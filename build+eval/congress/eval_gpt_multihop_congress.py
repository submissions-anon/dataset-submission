"""
eval_gpt_multihop_congress.py
==============================
GPT-5.2 + Gemini multi-hop retrieval for the Congressional Hearing ToT benchmark.

Per query:
  For each hop (1-4):
    1. GPT-5.2 generates a search query from (original description + notes from prior hops)
    2. Gemini retrieves top-K passages
    3. GPT-5.2 reads retrieved passages, selects plausible matches + writes notes

  Final:
    Pool A: GPT-selected candidates across all hops → GPT reranks
    Pool B: Retrieved but not selected → sorted by original query embedding score

Usage:
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...
    python eval_gpt_multihop_congress.py --benchmark-dir congress_corpus_data/beir_export/
    python eval_gpt_multihop_congress.py --benchmark-dir congress_corpus_data/beir_export/ --limit 5
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm
from openai import AsyncOpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark-dir", required=True)
parser.add_argument("--tag",           default="tot")
parser.add_argument("--model",         default="gpt-5.2")
parser.add_argument("--gemini-model",  default="gemini-embedding-2-preview")
parser.add_argument("--num-hops",      type=int, default=4)
parser.add_argument("--top-k-per-hop", type=int, default=25)
parser.add_argument("--max-candidates-per-hop", type=int, default=None)
parser.add_argument("--top-k-eval",    type=int, default=100)
parser.add_argument("--k-values",      default="10,50,100")
parser.add_argument("--ckpt",          default="checkpoints/gpt_multihop_congress_cache.jsonl")
parser.add_argument("--sleep",         type=float, default=0.5)
parser.add_argument("--concurrency",   type=int, default=10)
parser.add_argument("--seed",          type=int, default=42)
parser.add_argument("--limit",         type=int, default=None)
parser.add_argument("--dim",           type=int, default=3072)
parser.add_argument("--chunk-size",    type=int, default=500)
args = parser.parse_args()

k_values  = [int(k) for k in args.k_values.split(",")]
bench_dir = Path(args.benchmark_dir)
ckpt_path = Path(args.ckpt)
ckpt_path.parent.mkdir(parents=True, exist_ok=True)

# ── Clients ───────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")
openai_client = AsyncOpenAI(api_key=api_key, timeout=300)

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

You are an advanced retrieval system. You will be given a tip of tongue query                                                                                                                describing a users hazy memory of a specific moment from a US congressional hearing.                                                                                                         They wrote a vague description of what they remember. Given their 
description and any notes from previous search attempts, generate a search query 
that would match the actual hearing transcript.

Think about what the transcript would actually contain — speaker names, committee 
procedures, policy terms, specific phrases."""

QUERY_GEN_USER = """\
ORIGINAL DESCRIPTION:
{original_query}

NOTES FROM PREVIOUS HOPS:
{notes}

HOP: {hop_num} of {total_hops}

Generate a search query. Return ONLY JSON:
{{
  "search_query": "<query using transcript-language>",
  "rationale": "<what you're looking for this hop>"
}}"""

NOTE_EXTRACT_SYSTEM = """\
You are trying to find a specific congressional hearing moment that someone vaguely \
remembers. You will see their description and some candidate transcript passages.

Select any passages that could plausibly be the moment they're describing. \
Write notes about what you've learned so far — what matches, what doesn't, \
what to search for next."""

NOTE_EXTRACT_USER = """\
WHAT THEY REMEMBER:
{query}

PREVIOUS NOTES:
{previous_notes}

CANDIDATE PASSAGES (hop {hop_num}):
{candidates}

Analyze these passages. Return ONLY JSON:
{{
  "candidate_ids": ["id1", "id2", ...],
  "notes": "<what matched, what didn't, what to try next>",
  "summary": "<brief summary of this hop>"
}}

{selection_instruction}"""

RERANK_SYSTEM = """\
Rank these candidate transcript passages by how well they match the described \
moment. The person's description may have wrong details — focus on whether the \
overall dynamic and shape of the exchange matches.

Return ONLY JSON:
{{
  "ranked": [
    {{"candidate_num": <int>, "reason": "<brief>"}},
    ...
  ]
}}"""

RERANK_USER = """\
WHAT THEY REMEMBER:
{query}

CANDIDATE PASSAGES:
{candidates}

Rank all {k} candidates. Every number must appear exactly once."""

# ── Load benchmark ────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            d = json.loads(line)
            out[d['_id']] = d
    return out

print(f"\n{'='*70}")
print(f"  GPT-5.2 + Gemini {args.num_hops}-Hop Retrieval (Congress ToT)")
print(f"{'='*70}")

print("\n[1/5] Loading benchmark...")
corpus  = load_jsonl(bench_dir / "corpus.jsonl")
queries = load_jsonl(bench_dir / "queries.jsonl")

qrels = {}
with open(bench_dir / "qrels.tsv") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('query-id') or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) == 3:
            qid, cid, score = parts
        elif len(parts) == 4:
            qid, _, cid, score = parts
        else:
            continue
        qrels.setdefault(qid, {})[cid] = int(score)

query_witness = {}
query_memorability = {}
for qid, q in queries.items():
    meta = q.get('metadata', {})
    query_witness[qid] = meta.get('source_speaker', 'unknown').lower()
    query_memorability[qid] = meta.get('memorability', 0)

corpus_ids = list(corpus.keys())
query_ids  = [qid for qid in queries if qid in qrels]
if args.limit:
    query_ids = query_ids[:args.limit]

print(f"  Corpus: {len(corpus_ids):,} passages  |  Queries: {len(query_ids)}")

# ── Gemini embedding helpers ──────────────────────────────────────────────────

def embed_texts_gemini_sync(texts, task_type="RETRIEVAL_QUERY"):
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
            time.sleep(0.5)
            return np.array([e.values for e in result.embeddings], dtype=np.float32)
        except Exception as e:
            wait = 30 * (2 ** attempt)
            print(f"\n  [Gemini attempt {attempt+1}/6, retry in {wait}s] {e}")
            time.sleep(wait)
    raise RuntimeError("Gemini embedding failed after 6 attempts")


async def embed_texts_gemini(texts, task_type="RETRIEVAL_QUERY"):
    return await asyncio.to_thread(embed_texts_gemini_sync, texts, task_type)


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def retrieve_top_k(query_embedding, corpus_embeddings, corpus_ids, top_k, exclude_ids=None):
    scores = query_embedding @ corpus_embeddings.T
    scores = scores.flatten()
    if exclude_ids:
        exclude_set = set(exclude_ids)
        for i, did in enumerate(corpus_ids):
            if did in exclude_set:
                scores[i] = -np.inf
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus_ids[i], float(scores[i])) for i in top_indices if scores[i] > -np.inf]

# ── GPT helper ────────────────────────────────────────────────────────────────

async def call_gpt(system_prompt, user_prompt):
    for attempt in range(5):
        try:
            resp = await openai_client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=1.0,
                max_completion_tokens=128000,
            )
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                raise ValueError("Empty response")
            clean = raw.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = "\n".join(clean.split("\n")[:-1])
            return json.loads(clean.strip())
        except KeyboardInterrupt:
            raise
        except Exception as e:
            wait = 10 * (2 ** attempt)
            print(f"\n  [GPT attempt {attempt+1}/5, retry in {wait}s] {e}")
            await asyncio.sleep(wait)
    return None

# ── Load corpus embeddings ────────────────────────────────────────────────────

print("\n[2/5] Loading corpus embeddings...")
ckpt_dir = bench_dir / "gemini_cache"
model_slug_gemini = args.gemini_model.replace("/", "_").replace("-", "_")

chunk_starts = list(range(0, len(corpus_ids), args.chunk_size))
existing = {}
missing = []
for cs in chunk_starts:
    p = ckpt_dir / f"corpus_{model_slug_gemini}_chunk{cs}.npy"
    if p.exists():
        existing[cs] = p
    else:
        missing.append(cs)

print(f"  Chunks: {len(chunk_starts)} total, {len(existing)} cached, {len(missing)} missing")
if missing:
    raise SystemExit(f"Missing {len(missing)} corpus embedding chunks. Run eval_gemini_congress.py first.")

corpus_embeddings = normalize(np.concatenate([np.load(existing[cs]) for cs in chunk_starts], axis=0))
cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
print(f"  Shape: {corpus_embeddings.shape}")

# ── Multi-hop pipeline ───────────────────────────────────────────────────────

async def run_multihop_pipeline(query_id, query_text):
    accumulated_notes = ""
    all_candidates = {}   # did -> score (GPT-selected)
    all_retrieved = {}    # did -> score (everything retrieved)
    hop_details = []
    seen_docs = set()

    for hop_num in range(1, args.num_hops + 1):
        # Step 1: GPT generates search query
        query_gen_prompt = QUERY_GEN_USER.format(
            original_query=query_text,
            notes=accumulated_notes if accumulated_notes else "(First hop — no notes yet)",
            hop_num=hop_num,
            total_hops=args.num_hops,
        )
        gen_result = await call_gpt(QUERY_GEN_SYSTEM, query_gen_prompt)
        search_query = gen_result.get("search_query", query_text) if gen_result else query_text

        await asyncio.sleep(args.sleep)

        # Step 2: Gemini retrieves
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
            if did in corpus:
                candidate_lines.append(f"[{did}]\n{corpus[did]['text']}")
        candidates_str = "\n\n---\n\n".join(candidate_lines)

        # Step 4: GPT selects + extracts notes
        if args.max_candidates_per_hop:
            selection_instruction = f"Select up to {args.max_candidates_per_hop} passage IDs that could be the described moment."
        else:
            selection_instruction = "Select ALL passage IDs that could plausibly be the described moment."

        note_prompt = NOTE_EXTRACT_USER.format(
            query=query_text,
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
                if did_str in corpus:
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

    # Pool A: GPT-selected → rerank
    pool_a = set(all_candidates.keys())
    pool_b_dids = [did for did in all_retrieved if did not in pool_a]

    # Re-score Pool B with original query
    original_query_emb = normalize(await embed_texts_gemini([query_text], "RETRIEVAL_QUERY"))
    pool_b_scores = {}
    for did in pool_b_dids:
        idx = cid_to_idx.get(did, -1)
        if idx >= 0:
            pool_b_scores[did] = float(original_query_emb[0] @ corpus_embeddings[idx])

    # GPT reranks Pool A
    ranked = []
    if pool_a:
        pool_a_list = list(pool_a)
        num_to_did = {}
        candidate_lines = []
        for i, did in enumerate(pool_a_list):
            num = i + 1
            num_to_did[num] = did
            if did in corpus:
                candidate_lines.append(f"[{num}]\n{corpus[did]['text']}")

        candidates_str = "\n\n".join(candidate_lines)
        rerank_user = RERANK_USER.format(
            query=query_text,
            candidates=candidates_str,
            k=len(pool_a_list),
        )

        rerank_result = await call_gpt(RERANK_SYSTEM, rerank_user)

        if rerank_result and "ranked" in rerank_result:
            seen = set()
            for item in rerank_result["ranked"]:
                num = item.get("candidate_num")
                if num in num_to_did and num not in seen:
                    seen.add(num)
                    ranked.append({"did": num_to_did[num], "rank": len(ranked) + 1,
                                   "reason": item.get("reason", "")})
            for did in pool_a_list:
                if did not in {r["did"] for r in ranked}:
                    ranked.append({"did": did, "rank": len(ranked) + 1, "reason": "missed"})
        else:
            for did in sorted(pool_a_list, key=lambda d: -all_candidates.get(d, 0)):
                ranked.append({"did": did, "rank": len(ranked) + 1, "reason": "fallback"})

    # Append Pool B
    for did, score in sorted(pool_b_scores.items(), key=lambda x: -x[1]):
        ranked.append({"did": did, "rank": len(ranked) + 1, "score": score, "reason": "pool_b"})

    return {
        "ranked": ranked,
        "hops": hop_details,
        "candidates": list(pool_a),
        "notes": accumulated_notes,
        "pool_a_size": len(pool_a),
        "pool_b_size": len(pool_b_dids),
    }

# ── Load cache ────────────────────────────────────────────────────────────────

print("\n[3/5] Loading checkpoint...")
cached = {}
if ckpt_path.exists():
    with open(ckpt_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("ranked"):
                cached[rec["query_id"]] = rec
    print(f"  Loaded {len(cached)} cached results")
else:
    print(f"  Starting fresh")

# ── Run ───────────────────────────────────────────────────────────────────────

print(f"\n[4/5] Running {args.num_hops}-hop pipeline...")

results_list = []
pending_qids = []
for qid in query_ids:
    if qid in cached:
        results_list.append(cached[qid])
    else:
        pending_qids.append(qid)

print(f"  Cached:  {len(cached)}")
print(f"  Pending: {len(pending_qids)}")

cache_lock = asyncio.Lock()


async def process_one_query(qid, sem, cache_file, pbar):
    async with sem:
        try:
            query_text = queries[qid]["text"]
            result = await run_multihop_pipeline(qid, query_text)
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


async def run_all():
    sem = asyncio.Semaphore(args.concurrency)
    cache_file = open(ckpt_path, "a")
    pbar = tqdm(total=len(pending_qids), desc="Multi-hop")
    try:
        tasks = [process_one_query(qid, sem, cache_file, pbar) for qid in pending_qids]
        new_recs = await asyncio.gather(*tasks)
    finally:
        pbar.close()
        cache_file.close()
    return [r for r in new_recs if r is not None]


if pending_qids:
    new_recs = asyncio.run(run_all())
    results_list.extend(new_recs)

# ── Build run ─────────────────────────────────────────────────────────────────

run = {}
ranked_lists = {}
for rec in results_list:
    qid = rec["query_id"]
    ranked = rec.get("ranked", [])
    if not ranked:
        continue

    docs = {}
    total = len(ranked)
    ranked_order = []
    for item in ranked:
        did = item["did"]
        rank = item["rank"]
        docs[did] = float(total - rank + 1)
        ranked_order.append(did)

    run[qid] = docs
    ranked_lists[qid] = ranked_order

# ── Evaluate ──────────────────────────────────────────────────────────────────

print(f"\n[5/5] Evaluating...")

metrics = set()
for k in k_values:
    metrics.add(f"ndcg_cut_{k}")
    metrics.add(f"recall_{k}")
metrics.add("map")
metrics.add("recip_rank")

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
per_query = evaluator.evaluate(run)

ordered = (
    [f"ndcg_cut_{k}" for k in k_values] +
    [f"recall_{k}"   for k in k_values] +
    ["map", "recip_rank"]
)

agg = defaultdict(list)
for qid, ms in per_query.items():
    for m, v in ms.items():
        agg[m].append(v)

print(f"\n{'='*60}")
print(f"  {args.model} + Gemini {args.num_hops}-hop — Congress ToT")
print(f"  Top-K/hop: {args.top_k_per_hop}  |  Queries: {len(per_query)}")
print(f"{'='*60}")
print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8}")
print(f"  {'-'*52}")
for m in ordered:
    if m in agg:
        vals = agg[m]
        print(f"  {m:<25} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

# ── Per-memorability breakdown ────────────────────────────────────────────────

mem_ndcg10 = defaultdict(list)
for qid, md in per_query.items():
    m = query_memorability.get(qid, 0)
    mem_ndcg10[m].append(md.get("ndcg_cut_10", 0.0))

print(f"\n  {'─'*40}")
print(f"  Per-memorability nDCG@10")
print(f"  {'─'*40}")
print(f"  {'Memorability':<15} {'nDCG@10':>8} {'n':>5}")
print(f"  {'─'*40}")
for m in sorted(mem_ndcg10.keys(), reverse=True):
    vals = mem_ndcg10[m]
    print(f"  {m:<15} {np.mean(vals):>8.4f} {len(vals):>5}")

# ── Per-witness breakdown ─────────────────────────────────────────────────────

witness_ndcg10 = defaultdict(list)
witness_ndcg50 = defaultdict(list)
for qid, md in per_query.items():
    w = query_witness.get(qid, "unknown")
    witness_ndcg10[w].append(md.get("ndcg_cut_10", 0.0))
    witness_ndcg50[w].append(md.get("ndcg_cut_50", 0.0))

if witness_ndcg10:
    print(f"\n  {'─'*56}")
    print(f"  Per-witness nDCG@10")
    print(f"  {'─'*56}")
    print(f"  {'Witness':<30} {'nDCG@10':>8} {'nDCG@50':>8} {'n':>5}")
    print(f"  {'─'*56}")
    for w in sorted(witness_ndcg10, key=lambda a: -np.mean(witness_ndcg10[a])):
        v10 = witness_ndcg10[w]
        v50 = witness_ndcg50[w]
        print(f"  {w[:29]:<30} {np.mean(v10):>8.4f} {np.mean(v50):>8.4f} {len(v10):>5}")

print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""
gpt_slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
out_path = bench_dir / f"gpt_multihop_{args.num_hops}hop_{gpt_slug}{tag_suffix}_results.jsonl"

rows_out = []
for qid in query_ids:
    v = per_query.get(qid, {})
    rows_out.append({
        "query_id":      qid,
        "witness":       query_witness.get(qid, "unknown"),
        "memorability":  query_memorability.get(qid, 0),
        "ndcg@10":       round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50":       round(v.get("ndcg_cut_50", 0), 4),
        "recall@10":     round(v.get("recall_10",   0), 4),
        "recall@50":     round(v.get("recall_50",   0), 4),
        "recall@100":    round(v.get("recall_100",  0), 4),
        "map":           round(v.get("map",          0), 4),
        "mrr":           round(v.get("recip_rank",   0), 4),
        "ranked_list":   ranked_lists.get(qid, []),
    })
out_path.write_text("\n".join(json.dumps(r) for r in rows_out))

summary = {
    "model":           args.model,
    "retriever":       args.gemini_model,
    "algorithm":       f"multihop_{args.num_hops}",
    "tag":             args.tag,
    "top_k_per_hop":   args.top_k_per_hop,
    "num_hops":        args.num_hops,
    "corpus_size":     len(corpus_ids),
    "n_queries":       len(per_query),
    "metrics":         {m: round(float(np.mean(agg.get(m, [0]))), 4) for m in ordered},
    "per_memorability_ndcg10": {
        str(m): round(float(np.mean(v)), 4) for m, v in sorted(mem_ndcg10.items(), reverse=True)
    },
}
summary_path = bench_dir / f"gpt_multihop_{args.num_hops}hop_{gpt_slug}{tag_suffix}_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[+] Results → {out_path}")
print(f"[+] Summary → {summary_path}")
print(f"[+] Cache   → {ckpt_path}")
