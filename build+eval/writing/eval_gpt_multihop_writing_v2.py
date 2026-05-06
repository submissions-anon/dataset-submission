"""
eval_gpt_multihop_writing.py
============================
GPT-5.2 + Gemini 4-hop retrieval system for Writing Analogues (author attribution) benchmark.

Task: Given a text snippet, find other snippets written by the same author.

Pipeline per query:
  For each hop (1-4):
    1. GPT-5.2 generates a search query from (original_query + accumulated_notes)
    2. Gemini retrieves top-K snippets via dense embedding similarity
    3. GPT-5.2 reads retrieved snippets, selects pertinent ones + extracts notes

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
  python eval_gpt_multihop_writing.py --corpus-dir corpus
  python eval_gpt_multihop_writing.py --corpus-dir corpus --limit 5  # smoke test
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm
from openai import AsyncOpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark-dir", required=True,
                    help="Directory with corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json")
parser.add_argument("--corpus-dir", required=True,
                    help="Directory containing 1.txt, 2.txt, ... text files")
parser.add_argument("--gold-track", default="corpus_track.csv",
                    help="Path to v2 gold corpus_track.csv (snippet_id, author_name, post_title, post_url). "
                         "Used for per-author breakdown.")
parser.add_argument("--tag", default="v2",
                    help="Suffix for output filenames.")
parser.add_argument("--model", default="gpt-5.2", help="OpenAI model for reasoning")
parser.add_argument("--gemini-model", default="gemini-embedding-2-preview")
parser.add_argument("--num-hops", type=int, default=4, help="Number of retrieval hops")
parser.add_argument("--top-k-per-hop", type=int, default=25,
                    help="Snippets to retrieve per hop")
parser.add_argument("--max-candidates-per-hop", type=int, default=None,
                    help="Max candidate IDs GPT can select per hop")
parser.add_argument("--top-k-eval", type=int, default=100,
                    help="Total retrieval depth for evaluation")
parser.add_argument("--ckpt", default="checkpoints/gpt_multihop_writing_cache.jsonl")
parser.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds between phases within a query coroutine (rate-limit buffer)")
parser.add_argument("--concurrency", type=int, default=10,
                    help="Max concurrent query pipelines. Each pipeline does --num-hops sequential GPT calls.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--limit", type=int, default=None, help="Only run N queries (smoke test)")
parser.add_argument("--max-doc-chars", type=int, default=99999,
                    help="Max chars of snippet text shown to GPT per document")
parser.add_argument("--dim", type=int, default=3072, help="Gemini embedding dimension")
parser.add_argument("--chunk-size", type=int, default=500, help="Embedding chunk size")
args = parser.parse_args()

bench_dir = Path(args.benchmark_dir)
corpus_dir = Path(args.corpus_dir)
corpus_jsonl = bench_dir / "corpus.jsonl"
queries_jsonl = bench_dir / "queries.jsonl"
qrels_file = bench_dir / "qrels.tsv"
excluded_ids_file = bench_dir / "per_query_excluded_ids.json"

for f in [corpus_jsonl, queries_jsonl, qrels_file]:
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

# ── Clients ───────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")
openai_client = AsyncOpenAI(api_key=api_key)

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
You are an expert at author attribution - identifying texts written by the same author.

Given an original text snippet and notes from previous search iterations, generate a search
query that will help find other texts written by the same author.

The search query should:
1. Focus on distinctive stylistic features, vocabulary patterns, or thematic preferences
2. Capture the author's unique voice and writing mannerisms
3. Be different from previous search angles to maximize coverage

If this is the first hop, focus on the most distinctive stylistic markers. For later hops,
refine based on what patterns you've discovered."""

QUERY_GEN_USER = """\
ORIGINAL TEXT (find other texts by the same author):
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
You are an expert at author attribution - identifying texts written by the same author.

Your tasks:
1. Select text snippets that appear to be written by the SAME AUTHOR as the query text
2. Write brief notes about the stylistic patterns you observed

Look for: vocabulary choices, sentence structure, punctuation habits, thematic preferences,
tone, rhetorical devices, and other authorial fingerprints."""

NOTE_EXTRACT_USER = """\
QUERY TEXT (find other texts by this author):
{query}

PREVIOUS NOTES:
{previous_notes}

CANDIDATE TEXTS (from hop {hop_num}):
{candidates}

Analyze these texts. Return ONLY a JSON object:
{{
  "candidate_ids": ["id1", "id2", ...],
  "notes": "<observations about writing style, what patterns are MISSING>",
  "summary": "<brief summary of what was found in this hop>"
}}

{selection_instruction}
Add notes that could help find more texts by the same author in the next hop."""

RERANK_SYSTEM = """\
You are an expert at author attribution - identifying texts written by the same author.

Rank the candidates from most to least likely to be written by the same author as the query.
Focus on stylistic similarities: vocabulary, sentence structure, tone, thematic patterns.
Output exactly {k} items. Every candidate number must appear exactly once.

Return ONLY a JSON object:
{{
  "ranked": [
    {{"candidate_num": <int>, "reason": "<brief>"}},
    ...
  ]
}}"""

RERANK_USER = """\
QUERY TEXT (find other texts by this author):
{query}

CANDIDATE TEXTS:
{candidates}

Rank all {k} candidates by how likely they are written by the same author."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl_with_text(jsonl_path, text_dir):
    """Load JSONL and resolve text from .txt files."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                sid = d["_id"]
                txt_path = text_dir / f"{sid}.txt"
                if txt_path.exists():
                    d["text"] = txt_path.read_text(encoding="utf-8", errors="replace").strip()
                else:
                    d["text"] = ""
                out[sid] = d
    return out


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


async def call_gpt(system_prompt, user_prompt, parse_json=True):
    """Async GPT call with retry logic."""
    for attempt in range(5):
        try:
            resp = await openai_client.chat.completions.create(
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
            await asyncio.sleep(wait)
    return None


# ── Gemini Embedding ──────────────────────────────────────────────────────────

def embed_texts_gemini_sync(texts, task_type="RETRIEVAL_QUERY"):
    """Sync Gemini embed (called from a thread to avoid blocking the event loop)."""
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


async def embed_texts_gemini(texts, task_type="RETRIEVAL_QUERY"):
    """Async wrapper that runs the sync Gemini call in a thread."""
    return await asyncio.to_thread(embed_texts_gemini_sync, texts, task_type)


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
    results = [(corpus_ids[i], float(scores[i])) for i in top_indices if scores[i] > -np.inf]
    return results


# ── Multi-hop Pipeline ────────────────────────────────────────────────────────

async def run_multihop_pipeline(query_id, query_text, corpus_by_id, corpus_embeddings, corpus_ids, cid_to_idx, excluded_ids=None):
    """
    Async multi-hop pipeline for a single query. Each query's 4 hops are sequential
    internally, but multiple queries run concurrently via asyncio.gather + semaphore.

    Returns:
        dict with 'ranked', 'hops', 'candidates', 'notes', 'pool_a_size', 'pool_b_size'.
    """
    accumulated_notes = ""
    all_candidates = {}  # did -> best_score (GPT-selected)
    all_retrieved = {}   # did -> score (everything Gemini retrieved)
    hop_details = []
    seen_docs = set(excluded_ids) if excluded_ids else set()

    for hop_num in range(1, args.num_hops + 1):
        # Step 1: Generate search query
        query_gen_prompt = QUERY_GEN_USER.format(
            original_query=query_text,
            notes=accumulated_notes if accumulated_notes else "(No notes yet - this is the first hop)",
            hop_num=hop_num,
            total_hops=args.num_hops,
        )
        gen_result = await call_gpt(QUERY_GEN_SYSTEM, query_gen_prompt)
        if not gen_result:
            search_query = query_text
        else:
            search_query = gen_result.get("search_query", query_text)

        await asyncio.sleep(args.sleep)

        # Step 2: Embed search query and retrieve
        query_emb = normalize(await embed_texts_gemini([search_query], "RETRIEVAL_QUERY"))
        retrieved = retrieve_top_k(
            query_emb[0], corpus_embeddings, corpus_ids,
            top_k=args.top_k_per_hop,
            exclude_ids=list(seen_docs),
        )

        for did, score in retrieved:
            seen_docs.add(did)
            if did not in all_retrieved:
                all_retrieved[did] = score

        # Step 3: Format candidates for GPT
        candidate_lines = []
        for did, score in retrieved:
            if did in corpus_by_id:
                text = corpus_by_id[did]["text"][:args.max_doc_chars]
                if len(corpus_by_id[did]["text"]) > args.max_doc_chars:
                    text += "..."
                candidate_lines.append(f"[{did}] (score: {score:.4f})\n{text}")
        candidates_str = "\n\n---\n\n".join(candidate_lines)

        # Step 4: Extract notes and select candidates
        if args.max_candidates_per_hop:
            selection_instruction = f"Select up to {args.max_candidates_per_hop} candidate IDs that appear to be by the same author."
        else:
            selection_instruction = "Select ALL candidate IDs that appear to be by the same author (no limit)."

        note_prompt = NOTE_EXTRACT_USER.format(
            query=query_text[:3000],
            previous_notes=accumulated_notes if accumulated_notes else "(None yet)",
            hop_num=hop_num,
            candidates=candidates_str,
            selection_instruction=selection_instruction,
        )
        extract_result = await call_gpt(NOTE_EXTRACT_SYSTEM, note_prompt)

        if extract_result:
            selected_ids = extract_result.get("candidate_ids", [])
            new_notes = extract_result.get("notes", "")
            summary = extract_result.get("summary", "")

            for did in selected_ids:
                did_str = str(did)
                if did_str in corpus_by_id:
                    for rdid, rscore in retrieved:
                        if rdid == did_str:
                            if did_str not in all_candidates or rscore > all_candidates[did_str]:
                                all_candidates[did_str] = rscore
                            break

            if new_notes:
                accumulated_notes += f"\n\n[Hop {hop_num}]: {new_notes}"

            hop_details.append({
                "hop": hop_num,
                "search_query": search_query,
                "retrieved_count": len(retrieved),
                "selected_ids": selected_ids,
                "notes": new_notes,
                "summary": summary,
            })
        else:
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

        await asyncio.sleep(args.sleep)

    # Pool A: GPT-selected, Pool B: retrieved but not selected
    pool_a = set(all_candidates.keys())
    pool_b_dids = [did for did in all_retrieved if did not in pool_a]

    # Re-score Pool B with original query embedding
    original_query_emb = normalize(await embed_texts_gemini([query_text[:500]], "RETRIEVAL_QUERY"))
    pool_b_scores = {}
    for did in pool_b_dids:
        idx = cid_to_idx.get(did, -1)  # O(1) lookup vs O(n) index() in v1
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

        rerank_result = await call_gpt(rerank_system, rerank_user)

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
            for did in pool_a_list:
                if did not in {r["did"] for r in ranked}:
                    ranked.append({"did": did, "rank": len(ranked) + 1, "reason": "missed"})
        else:
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
print(f"  GPT-5.2 + Gemini {args.num_hops}-Hop Retrieval System (Writing Analogues)")
print(f"  Model: {args.model}  |  Gemini: {args.gemini_model}")
print(f"  Hops: {args.num_hops}  |  Top-K/hop: {args.top_k_per_hop}  |  Max candidates/hop: {args.max_candidates_per_hop}")
print(f"{'='*70}")

print("\n[1/5] Loading corpus...")
corpus_by_id = load_jsonl_with_text(corpus_jsonl, corpus_dir)
corpus_ids = list(corpus_by_id.keys())
print(f"  {len(corpus_ids):,} snippets")

print("\n[2/5] Loading queries & qrels...")
queries_by_id = load_jsonl_with_text(queries_jsonl, corpus_dir)
qrels = load_qrels(qrels_file)
query_ids = [qid for qid in queries_by_id if qid in qrels]

# Load per-query excluded IDs (same-post snippets shouldn't be retrieved)
per_query_excluded = {}
if excluded_ids_file.exists():
    with open(excluded_ids_file) as f:
        per_query_excluded = json.load(f)
    per_query_excluded = {qid: [str(x) for x in ids] for qid, ids in per_query_excluded.items()}
    print(f"  Loaded per-query exclusions for {len(per_query_excluded)} queries")

if args.limit:
    query_ids = query_ids[:args.limit]
    print(f"  Smoke test: limiting to {args.limit} queries")

print(f"  {len(query_ids)} queries with qrels")
counts = [len(v) for v in qrels.values()]
avg_rel = float(np.mean(counts))
print(f"  Avg rel/query: {avg_rel:.1f}  |  Min/Max: {min(counts)}/{max(counts)}")

# Load per-author metadata for breakdowns (v2 gold CSV)
import csv as _csv
from collections import defaultdict as _defaultdict
snippet_author = {}

csv_path = Path(args.gold_track)
if csv_path.exists():
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        reader.fieldnames = [fn.strip() for fn in reader.fieldnames or []]
        for row in reader:
            sid = (row.get("snippet_id") or "").strip()
            author = (row.get("author_name") or "").strip()
            if sid and author:
                snippet_author[sid] = author
    print(f"  Loaded author metadata for {len(snippet_author)} gold snippets")

print("\n[3/5] Loading corpus embeddings...")
ckpt_dir = bench_dir / "gemini_cache"
model_slug = args.gemini_model.replace("/", "_").replace("-", "_")

# Load chunked embeddings
chunk_files = sorted(ckpt_dir.glob(f"corpus_{model_slug}_chunk*.npy"))
if chunk_files:
    print(f"  Loading {len(chunk_files)} embedding chunks...")
    chunks = []
    for cf in chunk_files:
        chunks.append(np.load(cf))
    corpus_embeddings = normalize(np.concatenate(chunks, axis=0))
    print(f"  Embeddings shape: {corpus_embeddings.shape}")
else:
    raise FileNotFoundError(
        f"Corpus embeddings not found in {ckpt_dir}. "
        f"Run eval_gemini.py first to generate them."
    )

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

# Build cid_to_idx ONCE (was being built per-query inside the pipeline as O(n) list.index in v1)
cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

# Separate cached from pending
results_list = []
pending_qids = []
for qid in query_ids:
    if qid in cached:
        results_list.append(cached[qid])
    else:
        pending_qids.append(qid)

print(f"  Cached:  {len(cached)}")
print(f"  Pending: {len(pending_qids)}")
print(f"  Concurrency: {args.concurrency} queries in flight at a time")
print(f"  Per query: {args.num_hops} sequential hops × ~3 GPT calls + 1 final rerank")

# Lock around cache file so concurrent coroutines don't interleave writes
cache_lock = asyncio.Lock()


async def process_one_query(qid, sem, cache_file, pbar):
    """Run the full multihop pipeline for one query, semaphore-bounded."""
    async with sem:
        try:
            query_text = queries_by_id[qid]["text"]
            excluded_ids = per_query_excluded.get(qid, [])
            result = await run_multihop_pipeline(
                qid, query_text,
                corpus_by_id, corpus_embeddings, corpus_ids, cid_to_idx,
                excluded_ids=excluded_ids,
            )
            rec = {
                "query_id":   qid,
                "ranked":     result["ranked"],
                "hops":       result["hops"],
                "candidates": result["candidates"],
                "notes":      result.get("notes", ""),
            }
            async with cache_lock:
                cache_file.write(json.dumps(rec) + "\n")
                cache_file.flush()
            pbar.update(1)
            return rec
        except Exception as e:
            pbar.update(1)
            print(f"\n  [query {qid} failed] {e}")
            return None


async def run_all_queries(pending_qids):
    sem = asyncio.Semaphore(args.concurrency)
    cache_file = open(ckpt_path, "a")
    pbar = tqdm(total=len(pending_qids), desc="Multi-hop queries")
    try:
        tasks = [process_one_query(qid, sem, cache_file, pbar) for qid in pending_qids]
        new_recs = await asyncio.gather(*tasks)
    finally:
        pbar.close()
        cache_file.close()
    return [r for r in new_recs if r is not None]


if pending_qids:
    new_recs = asyncio.run(run_all_queries(pending_qids))
    results_list.extend(new_recs)

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
print(f"  {args.model} + Gemini  --  {args.num_hops}-hop retrieval (Writing Analogues)")
print(f"  Top-K/hop: {args.top_k_per_hop}  |  Max candidates/hop: {args.max_candidates_per_hop}")
print(f"  Corpus: {len(corpus_ids):,} snippets  |  Queries: {len(query_ids)}")
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

# ── Per-author breakdown ──────────────────────────────────────────────────────

author_ndcg10 = _defaultdict(list)
author_ndcg50 = _defaultdict(list)

for qid in query_ids:
    md = results.get(qid, {})
    a  = snippet_author.get(qid, "unknown")
    author_ndcg10[a].append(md.get("ndcg_cut_10", 0.0))
    author_ndcg50[a].append(md.get("ndcg_cut_50", 0.0))

if snippet_author:
    print(f"\n  {'─'*56}")
    print(f"  Per-author nDCG@10")
    print(f"  {'─'*56}")
    print(f"  {'Author':<30} {'nDCG@10':>8} {'nDCG@50':>8} {'n':>5}")
    print(f"  {'─'*56}")
    for author in sorted(author_ndcg10, key=lambda a: -np.mean(author_ndcg10[a])):
        v10 = author_ndcg10[author]
        v50 = author_ndcg50[author]
        print(f"  {author[:29]:<30} {np.mean(v10):>8.4f} {np.mean(v50):>8.4f} {len(v10):>5}")

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

tag_suffix = f"_{args.tag}" if args.tag else ""
model_slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
out_path = bench_dir / f"results_gpt_multihop_{args.num_hops}hop_{model_slug}{tag_suffix}.jsonl"

# Build lookup from results_list for full ranked info (includes reasons, etc.)
results_by_qid = {rec["query_id"]: rec for rec in results_list}

rows = []
for qid in query_ids:
    v = results.get(qid, {})
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks = [rank_map[d] for d in rel_docs if d in rank_map]

    # Get full ranked list from pipeline results (includes reason field)
    full_ranked = results_by_qid.get(qid, {}).get("ranked", [])

    rows.append({
        "query_id": qid,
        "query_text": queries_by_id[qid]["text"][:120],
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
        "ranked": full_ranked,  # Full reranking list with reasons
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Per-query results -> {out_path}")
print(f"[+] Cache             -> {ckpt_path}  (safe to resume from)")

# ── Final Summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  SUMMARY: {args.model} + Gemini {args.num_hops}-hop (Writing Analogues)")
print(f"{'='*70}")
print(f"  NDCG@10:    {mean('ndcg_cut_10'):.4f}")
print(f"  NDCG@50:    {mean('ndcg_cut_50'):.4f}")
print(f"  Recall@10:  {mean('recall_10'):.4f}")
print(f"  Recall@50:  {mean('recall_50'):.4f}")
print(f"  Recall@100: {mean('recall_100'):.4f}")
print(f"  MRR:        {mean('recip_rank'):.4f}")
print(f"{'='*70}")
