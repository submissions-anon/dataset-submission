"""
eval_gpt_multihop_twitter.py
============================
GPT-5.2 + Gemini 4-hop retrieval system for Twitter Descriptive-IR benchmark.

Pipeline per query:
  For each hop (1-4):
    1. GPT-5.2 generates a search query from (original_query + accumulated_notes)
    2. Gemini retrieves top-K tweets via dense embedding similarity
    3. GPT-5.2 reads retrieved tweets, selects pertinent ones + extracts notes

  Final:
    Pool A: GPT-selected candidates across hops
    Pool B: Retrieved but not selected (re-scored with original query)
    GPT reranks Pool A -> top positions
    Pool B sorted by original query score -> remaining positions

Requirements:
  pip install google-genai pytrec_eval tqdm numpy openai

Usage:
  export OPENAI_API_KEY=...
  export GEMINI_API_KEY=...
  python eval_gpt_multihop_twitter.py
  python eval_gpt_multihop_twitter.py --limit 5  # smoke test
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm
from openai import OpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset")
parser.add_argument("--model", default="gpt-5.2", help="OpenAI model for reasoning")
parser.add_argument("--gemini-model", default="gemini-embedding-2-preview")
parser.add_argument("--num-hops", type=int, default=4, help="Number of retrieval hops")
parser.add_argument("--top-k-per-hop", type=int, default=25,
                    help="Tweets to retrieve per hop")
parser.add_argument("--max-candidates-per-hop", type=int, default=None,
                    help="Max candidate IDs GPT can select per hop")
parser.add_argument("--top-k-eval", type=int, default=100,
                    help="Total retrieval depth for evaluation")
parser.add_argument("--ckpt", default="checkpoints/gpt_multihop_twitter_cache.jsonl")
parser.add_argument("--sleep", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--limit", type=int, default=None, help="Only run N queries (smoke test)")
parser.add_argument("--max-doc-chars", type=int, default=99999,
                    help="Max chars of tweet text shown to GPT per document")
parser.add_argument("--dim", type=int, default=3072, help="Gemini embedding dimension")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
corpus_file = dataset_dir / "corpus_full.jsonl"
queries_file = dataset_dir / "queries.jsonl"
qrels_file = dataset_dir / "qrels.tsv"

for f in [corpus_file, queries_file, qrels_file]:
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

# ── Clients ───────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")
openai_client = OpenAI(api_key=api_key)

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise SystemExit("Set GEMINI_API_KEY env var")

try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    raise ImportError("pip install google-genai")
gemini_client = genai.Client(api_key=gemini_api_key)

# ── Prompts ───────────────────────────────────────────────────────────────────

QUERY_GEN_SYSTEM = """\
You are an expert at iterative information retrieval for finding tweets with specific implicit themes.

Given an original query describing a theme to find in tweets, and notes from previous search
iterations, generate a focused search query that will help find more relevant tweets.

The search query should:
1. Target specific aspects of the theme not yet well-covered
2. Use concrete keywords likely to appear in relevant tweets
3. Be different from previous search angles to maximize coverage

If this is the first hop (no notes yet), generate an initial search query based on the
core theme. For later hops, refine based on what's been learned."""

QUERY_GEN_USER = """\
ORIGINAL QUERY (theme to find):
{original_query}

ACCUMULATED NOTES FROM PREVIOUS HOPS:
{notes}

HOP NUMBER: {hop_num} of {total_hops}

Generate a search query for this hop. Return ONLY a JSON object:
{{
  "search_query": "<your search query>",
  "rationale": "<brief explanation of search strategy>"
}}"""

NOTE_EXTRACT_SYSTEM = """\
You are an expert at analyzing tweets for implicit themes and sentiments.

Your tasks:
1. Select tweets that exhibit the theme described in the query
2. Write brief notes about what you observed (to inform the next search)

A tweet is relevant ONLY IF it implicitly expresses the theme described in the query."""

NOTE_EXTRACT_USER = """\
QUERY (theme to find):
{query}

PREVIOUS NOTES:
{previous_notes}

CANDIDATE TWEETS (from hop {hop_num}):
{candidates}

Analyze these tweets. Return ONLY a JSON object:
{{
  "candidate_ids": ["id1", "id2", ...],
  "notes": "<observations about patterns seen, what's MISSING, what to search for next>",
  "summary": "<brief summary of what was found in this hop>"
}}

{selection_instruction}
Add notes that could help find more relevant tweets in the next hop."""

RERANK_SYSTEM = """\
You are an expert at identifying implicit themes in tweets.

Rank the candidates from most to least relevant to the theme in the query.
Output exactly {k} items. Every candidate number must appear exactly once.

Return ONLY a JSON object:
{{
  "ranked": [
    {{"candidate_num": <int>, "reason": "<brief>"}},
    ...
  ]
}}"""

RERANK_USER = """\
QUERY (theme to find):
{query}

CANDIDATE TWEETS:
{candidates}

Rank all {k} candidates."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_corpus(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                queries[q["_id"]] = q["text"]
    return queries


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, did, score = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def call_gpt(system_prompt, user_prompt, parse_json=True):
    """Call GPT with retry logic."""
    for attempt in range(5):
        try:
            resp = openai_client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                max_completion_tokens=128000,
            )
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                raise ValueError("Empty response")

            if parse_json:
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = "\n".join(clean.split("\n")[1:])
                if clean.endswith("```"):
                    clean = "\n".join(clean.split("\n")[:-1])
                return json.loads(clean.strip())
            return raw

        except KeyboardInterrupt:
            raise
        except Exception as e:
            wait = 10 * (2 ** attempt)
            print(f"\n  [GPT attempt {attempt+1}/5, retry in {wait}s] {e}")
            time.sleep(wait)
    return None


# ── Gemini Embedding ──────────────────────────────────────────────────────────

def embed_texts_gemini(texts, task_type="RETRIEVAL_QUERY"):
    """Embed texts with Gemini."""
    for attempt in range(6):
        try:
            cfg = gtypes.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=args.dim,
            )
            result = gemini_client.models.embed_content(
                model=args.gemini_model,
                contents=texts,
                config=cfg,
            )
            time.sleep(0.5)  # rate limit
            return np.array([e.values for e in result.embeddings], dtype=np.float32)
        except Exception as e:
            wait = 30 * (2 ** attempt)
            print(f"\n  [Gemini attempt {attempt+1}/6, retry in {wait}s] {e}")
            time.sleep(wait)
    raise RuntimeError("Gemini embedding failed after 6 attempts")


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def retrieve_top_k(query_embedding, corpus_embeddings, corpus_ids, top_k, exclude_ids=None):
    """Retrieve top-K documents by cosine similarity."""
    scores = query_embedding @ corpus_embeddings.T
    scores = scores.flatten()

    # Create mask for excluded IDs
    if exclude_ids:
        exclude_set = set(exclude_ids)
        for i, did in enumerate(corpus_ids):
            if did in exclude_set:
                scores[i] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(corpus_ids[i], float(scores[i])) for i in top_indices]
    return results


# ── Multi-hop Pipeline ────────────────────────────────────────────────────────

def run_multihop_pipeline(query_id, query_text, corpus_by_id, corpus_embeddings, corpus_ids):
    """
    Run the 4-hop retrieval pipeline for a single query.

    Returns:
        dict with 'ranked' (final ranking), 'hops' (per-hop details), 'candidates' (all candidates)
    """
    accumulated_notes = ""
    all_candidates = {}  # did -> best_score (GPT-selected)
    all_retrieved = {}   # did -> score (everything Gemini retrieved)
    hop_details = []
    seen_docs = set()

    for hop_num in range(1, args.num_hops + 1):
        print(f"\n    [Hop {hop_num}/{args.num_hops}]")

        # Step 1: Generate search query
        query_gen_prompt = QUERY_GEN_USER.format(
            original_query=query_text,
            notes=accumulated_notes if accumulated_notes else "(No notes yet - this is the first hop)",
            hop_num=hop_num,
            total_hops=args.num_hops,
        )
        gen_result = call_gpt(QUERY_GEN_SYSTEM, query_gen_prompt)
        if not gen_result:
            print(f"      Query generation failed, using original query")
            search_query = query_text
        else:
            search_query = gen_result.get("search_query", query_text)
            print(f"      Search query: {search_query[:80]}...")

        time.sleep(args.sleep)

        # Step 2: Embed search query and retrieve
        query_emb = normalize(embed_texts_gemini([search_query], "RETRIEVAL_QUERY"))
        retrieved = retrieve_top_k(
            query_emb[0], corpus_embeddings, corpus_ids,
            top_k=args.top_k_per_hop,
            exclude_ids=list(seen_docs)  # Don't re-retrieve already seen docs
        )
        print(f"      Retrieved {len(retrieved)} tweets")

        # Track seen docs and all retrieved
        for did, score in retrieved:
            seen_docs.add(did)
            if did not in all_retrieved:
                all_retrieved[did] = score

        # Step 3: Format candidates for GPT
        candidate_lines = []
        for i, (did, score) in enumerate(retrieved):
            if did in corpus_by_id:
                text = corpus_by_id[did]["text"][:args.max_doc_chars]
                if len(corpus_by_id[did]["text"]) > args.max_doc_chars:
                    text += "..."
                candidate_lines.append(f"[{did}] (score: {score:.4f})\n{text}")

        candidates_str = "\n\n---\n\n".join(candidate_lines)

        # Step 4: Extract notes and select candidates
        if args.max_candidates_per_hop:
            selection_instruction = f"Select up to {args.max_candidates_per_hop} candidate IDs that best match the theme."
        else:
            selection_instruction = "Select ALL candidate IDs that match the theme (no limit)."

        note_prompt = NOTE_EXTRACT_USER.format(
            query=query_text,
            previous_notes=accumulated_notes if accumulated_notes else "(None yet)",
            hop_num=hop_num,
            candidates=candidates_str,
            selection_instruction=selection_instruction,
        )
        extract_result = call_gpt(NOTE_EXTRACT_SYSTEM, note_prompt)

        if extract_result:
            selected_ids = extract_result.get("candidate_ids", [])
            new_notes = extract_result.get("notes", "")
            summary = extract_result.get("summary", "")

            # Add selected candidates to pool
            for did in selected_ids:
                if did in corpus_by_id:
                    # Use retrieval score as initial score
                    for rdid, rscore in retrieved:
                        if rdid == did:
                            if did not in all_candidates or rscore > all_candidates[did]:
                                all_candidates[did] = rscore
                            break

            # Accumulate notes
            if new_notes:
                accumulated_notes += f"\n\n[Hop {hop_num}]: {new_notes}"

            print(f"      Selected {len(selected_ids)} candidates, notes: {len(new_notes)} chars")

            hop_details.append({
                "hop": hop_num,
                "search_query": search_query,
                "retrieved_count": len(retrieved),
                "selected_ids": selected_ids,
                "notes": new_notes,
                "summary": summary,
            })
        else:
            print(f"      Note extraction failed, adding all retrieved as candidates")
            for did, score in retrieved:
                if did not in all_candidates or score > all_candidates[did]:
                    all_candidates[did] = score
            hop_details.append({
                "hop": hop_num,
                "search_query": search_query,
                "retrieved_count": len(retrieved),
                "selected_ids": [did for did, _ in retrieved],
                "notes": "",
                "summary": "Extraction failed",
            })

        time.sleep(args.sleep)

    # Pool A: GPT-selected, Pool B: retrieved but not selected
    pool_a = set(all_candidates.keys())
    pool_b_dids = [did for did in all_retrieved if did not in pool_a]

    print(f"\n    [Pool A] {len(pool_a)} GPT-selected")
    print(f"    [Pool B] {len(pool_b_dids)} retrieved but not selected")

    # Re-score Pool B with original query embedding
    original_query_emb = normalize(embed_texts_gemini([query_text], "RETRIEVAL_QUERY"))
    pool_b_scores = {}
    for did in pool_b_dids:
        idx = corpus_ids.index(did) if did in corpus_ids else -1
        if idx >= 0:
            score = float(original_query_emb[0] @ corpus_embeddings[idx])
            pool_b_scores[did] = score

    # GPT reranks Pool A
    ranked = []
    if len(pool_a) > 0:
        pool_a_list = list(pool_a)
        candidate_lines = []
        num_to_did = {}
        for i, did in enumerate(pool_a_list):
            num = i + 1
            num_to_did[num] = did
            if did in corpus_by_id:
                text = corpus_by_id[did]["text"][:args.max_doc_chars]
                if len(corpus_by_id[did]["text"]) > args.max_doc_chars:
                    text += "..."
                candidate_lines.append(f"[{num}] {text}")

        candidates_str = "\n\n".join(candidate_lines)
        rerank_system = RERANK_SYSTEM.format(k=len(pool_a_list))
        rerank_user = RERANK_USER.format(
            query=query_text,
            candidates=candidates_str,
            k=len(pool_a_list),
        )

        print(f"    [Rerank] GPT reranking {len(pool_a_list)} Pool A candidates...")
        rerank_result = call_gpt(rerank_system, rerank_user)

        if rerank_result and "ranked" in rerank_result:
            seen = set()
            for item in rerank_result["ranked"]:
                num = item.get("candidate_num")
                if num in num_to_did and num not in seen:
                    seen.add(num)
                    ranked.append({
                        "did": num_to_did[num],
                        "rank": len(ranked) + 1,
                        "reason": item.get("reason", ""),
                    })
            # Add any missed from Pool A
            for did in pool_a_list:
                if did not in {r["did"] for r in ranked}:
                    ranked.append({"did": did, "rank": len(ranked) + 1, "reason": "missed"})
        else:
            print(f"    [Rerank] Failed, using retrieval scores for Pool A")
            for did in sorted(pool_a_list, key=lambda d: -all_candidates.get(d, 0)):
                ranked.append({"did": did, "rank": len(ranked) + 1, "reason": "fallback"})

    # Append Pool B sorted by original query score
    pool_b_sorted = sorted(pool_b_scores.items(), key=lambda x: -x[1])
    for did, score in pool_b_sorted:
        ranked.append({
            "did": did,
            "rank": len(ranked) + 1,
            "score": score,
            "reason": "pool_b",
        })

    print(f"    [Final] {len(ranked)} total ranked")

    return {
        "ranked": ranked,
        "hops": hop_details,
        "candidates": list(pool_a),
        "notes": accumulated_notes,
        "pool_a_size": len(pool_a),
        "pool_b_size": len(pool_b_dids),
    }


# ── Load Data ─────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  GPT-5.2 + Gemini {args.num_hops}-Hop Retrieval System (Twitter)")
print(f"  Model: {args.model}  |  Gemini: {args.gemini_model}")
print(f"  Hops: {args.num_hops}  |  Top-K/hop: {args.top_k_per_hop}  |  Max candidates/hop: {args.max_candidates_per_hop}")
print(f"{'='*70}")

print("\n[1/5] Loading corpus...")
corpus_by_id = {}
corpus_ids = []
for doc in tqdm(iter_corpus(corpus_file), desc="  Loading", unit="tweet"):
    corpus_by_id[doc["_id"]] = {"text": doc["text"]}
    corpus_ids.append(doc["_id"])
print(f"  {len(corpus_ids):,} tweets")

print("\n[2/5] Loading queries & qrels...")
queries = load_queries(queries_file)
qrels = load_qrels(qrels_file)
query_ids = [qid for qid in queries if qid in qrels]
query_texts = {qid: queries[qid] for qid in query_ids}

if args.limit:
    query_ids = query_ids[:args.limit]
    print(f"  Smoke test: limiting to {args.limit} queries")

print(f"  {len(query_ids)} queries with qrels")
counts = [len(v) for v in qrels.values()]
avg_rel = float(np.mean(counts))
print(f"  Avg rel/query: {avg_rel:.1f}  |  Min/Max: {min(counts)}/{max(counts)}")

print("\n[3/5] Loading corpus embeddings...")
ckpt_dir = Path("eval_cache_gemini")
model_slug = args.gemini_model.replace("/", "_").replace("-", "_")
corpus_emb_path = ckpt_dir / f"corpus_{model_slug}_full_tweet.npy"

if corpus_emb_path.exists():
    print(f"  Loading cached embeddings from {corpus_emb_path}")
    corpus_embeddings = normalize(np.load(corpus_emb_path))
else:
    raise FileNotFoundError(
        f"Corpus embeddings not found at {corpus_emb_path}. "
        f"Run eval_twitter_gemini.py first to generate them."
    )
print(f"  Embeddings shape: {corpus_embeddings.shape}")

# ── Load checkpoint ───────────────────────────────────────────────────────────

print("\n[4/5] Loading checkpoint...")
ckpt_path = Path(args.ckpt)
ckpt_path.parent.mkdir(exist_ok=True)
cached = {}
if ckpt_path.exists():
    with open(ckpt_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("ranked"):
                cached[rec["query_id"]] = rec
    print(f"  Loaded {len(cached)} cached results from {ckpt_path}")
else:
    print(f"  No checkpoint found, starting fresh")

# ── Run pipeline ──────────────────────────────────────────────────────────────

print(f"\n[5/5] Running {args.num_hops}-hop pipeline...")

results_list = []
cache_file = open(ckpt_path, "a")

for qid in tqdm(query_ids, desc="Processing queries"):
    if qid in cached:
        results_list.append(cached[qid])
        continue

    print(f"\n  Query {qid}: {query_texts[qid][:60]}...")

    result = run_multihop_pipeline(
        qid, query_texts[qid],
        corpus_by_id, corpus_embeddings, corpus_ids
    )

    rec = {
        "query_id": qid,
        "ranked": result["ranked"],
        "hops": result["hops"],
        "candidates": result["candidates"],
        "notes": result.get("notes", ""),
    }
    results_list.append(rec)
    cache_file.write(json.dumps(rec) + "\n")
    cache_file.flush()
    time.sleep(args.sleep)

cache_file.close()

# ── Build run dict ────────────────────────────────────────────────────────────

run = {}
for rec in results_list:
    qid = rec["query_id"]
    ranked = rec.get("ranked", [])
    candidates = rec.get("candidates", [])

    docs = {}
    total = max(len(ranked), len(candidates), 1)

    for item in ranked:
        did = item["did"]
        rank = item["rank"]
        docs[did] = float(total - rank + 1)

    run[qid] = docs

# ── Evaluate ──────────────────────────────────────────────────────────────────

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {
        "ndcg_cut.10,50,100",
        "recall.10,50,100",
        "recip_rank",
        "success.1,5,10",
    }
)
results = evaluator.evaluate(run)


def mean(key):
    vals = [v.get(key, 0.0) for v in results.values()]
    return float(np.mean(vals)) if vals else 0.0


best_ranks = []
for qid in query_ids:
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks = [rank_map[d] for d in rel_docs if d in rank_map]
    if ranks:
        best_ranks.append(min(ranks))
best_ranks_arr = np.array(best_ranks) if best_ranks else np.array([])

print(f"\n{'='*70}")
print(f"  {args.model} + Gemini  —  {args.num_hops}-hop retrieval (Twitter)")
print(f"  Top-K/hop: {args.top_k_per_hop}  |  Max candidates/hop: {args.max_candidates_per_hop}")
print(f"  Corpus: {len(corpus_ids):,} tweets  |  Queries: {len(query_ids)}")
print(f"{'='*70}")
print(f"  {'Metric':<22} {'Score':>8}")
print(f"  {'-'*32}")
print(f"  {'MRR':<22} {mean('recip_rank'):>8.4f}")
print(f"  {'NDCG@10':<22} {mean('ndcg_cut_10'):>8.4f}  <- primary")
print(f"  {'NDCG@50':<22} {mean('ndcg_cut_50'):>8.4f}")
print(f"  {'NDCG@100':<22} {mean('ndcg_cut_100'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Recall@10':<22} {mean('recall_10'):>8.4f}")
print(f"  {'Recall@50':<22} {mean('recall_50'):>8.4f}")
print(f"  {'Recall@100':<22} {mean('recall_100'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Success@1':<22} {mean('success_1'):>8.4f}")
print(f"  {'Success@5':<22} {mean('success_5'):>8.4f}")
print(f"  {'Success@10':<22} {mean('success_10'):>8.4f}")
if len(best_ranks_arr):
    print(f"  {'-'*32}")
    print(f"  Best-relevant-doc rank:")
    for thresh in [1, 5, 10, 50, 100]:
        n = int((best_ranks_arr <= thresh).sum())
        print(f"    Top-{thresh:<5} {n:>3}/{len(query_ids)}  ({n/len(query_ids):.0%})")
print(f"{'='*70}")

# ── Per-query breakdown ───────────────────────────────────────────────────────

print(f"\n  {'qid':<8} {'n_rel':>5}  {'NDCG@10':>8}  {'R@10':>7}  {'R@50':>7}  {'R@100':>7}  {'n_cand':>7}  {'best':>6}")
print(f"  {'-'*70}")
for qid in query_ids:
    v = results.get(qid, {})
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks = [rank_map[d] for d in rel_docs if d in rank_map]
    best = min(ranks) if ranks else -1
    n_cand = len(run.get(qid, {}))
    print(f"  {qid:<8} {len(qrels[qid]):>5}  "
          f"{v.get('ndcg_cut_10', 0):>8.4f}  "
          f"{v.get('recall_10', 0):>7.4f}  "
          f"{v.get('recall_50', 0):>7.4f}  "
          f"{v.get('recall_100', 0):>7.4f}  "
          f"{n_cand:>7}  "
          f"{best:>6}")

# ── Save ──────────────────────────────────────────────────────────────────────

model_slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
out_path = dataset_dir / f"results_gpt_multihop_{args.num_hops}hop_{model_slug}.jsonl"

rows = []
for qid in query_ids:
    v = results.get(qid, {})
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks = [rank_map[d] for d in rel_docs if d in rank_map]
    rows.append({
        "query_id": qid,
        "query_text": queries[qid][:120],
        "best_rank": min(ranks) if ranks else None,
        "n_relevant": len(rel_docs),
        "n_candidates": len(run.get(qid, {})),
        "mrr": round(v.get("recip_rank", 0), 4),
        "ndcg@10": round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50": round(v.get("ndcg_cut_50", 0), 4),
        "ndcg@100": round(v.get("ndcg_cut_100", 0), 4),
        "recall@10": round(v.get("recall_10", 0), 4),
        "recall@50": round(v.get("recall_50", 0), 4),
        "recall@100": round(v.get("recall_100", 0), 4),
        "success@1": round(v.get("success_1", 0), 4),
        "success@5": round(v.get("success_5", 0), 4),
        "success@10": round(v.get("success_10", 0), 4),
        "top10": ranked[:10],
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Per-query results -> {out_path}")
print(f"[+] Cache             -> {ckpt_path}  (safe to resume from)")

# ── Final Summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  SUMMARY: {args.model} + Gemini {args.num_hops}-hop (Twitter)")
print(f"{'='*70}")
print(f"  NDCG@10:    {mean('ndcg_cut_10'):.4f}")
print(f"  NDCG@50:    {mean('ndcg_cut_50'):.4f}")
print(f"  Recall@10:  {mean('recall_10'):.4f}")
print(f"  Recall@50:  {mean('recall_50'):.4f}")
print(f"  Recall@100: {mean('recall_100'):.4f}")
print(f"  MRR:        {mean('recip_rank'):.4f}")
print(f"{'='*70}")
