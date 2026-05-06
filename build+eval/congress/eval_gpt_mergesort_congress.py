"""
eval_gpt5_mergesort_congress.py
================================
MergeSortInter GPT reranking for the Congressional Hearing ToT benchmark.

Algorithm:
  1. Union top-N from Gemini + Qwen4B + Qwen0.6B dense retrieval results
  2. Gold-inject, shuffle
  3. Chunk into batches, sort each batch via listwise LLM call (PARALLEL)
  4. Extract top-k from each batch
  5. Shuffle and sort all top-k together
  6. Interleave remaining candidates round-robin
  7. Evaluate with pytrec_eval

Usage:
    export OPENAI_API_KEY=...
    python eval_gpt5_mergesort_congress.py --benchmark-dir congress_corpus_data/beir_export/
    python eval_gpt5_mergesort_congress.py --benchmark-dir congress_corpus_data/beir_export/ --limit 5
"""

import argparse
import asyncio
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
from openai import AsyncOpenAI
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark-dir",   required=True,
                    help="Directory with corpus.jsonl, queries.jsonl, qrels.tsv")
parser.add_argument("--tag",             default="tot")
parser.add_argument("--model",           default="gpt-5.2")

# Dense retrieval result files (relative to --benchmark-dir)
parser.add_argument("--gemini",          default="")
parser.add_argument("--qwen4b",          default="")
parser.add_argument("--qwen06b",         default="")

# Pool sizes per model
parser.add_argument("--gemini-top-n",    type=int, default=55)
parser.add_argument("--qwen4b-top-n",    type=int, default=25)
parser.add_argument("--qwen06b-top-n",   type=int, default=25)

parser.add_argument("--batch-size",      type=int, default=20)
parser.add_argument("--top-k",           type=int, default=4)
parser.add_argument("--concurrency",     type=int, default=10)
parser.add_argument("--cache",           default=None)
parser.add_argument("--sleep",           type=float, default=0.3)
parser.add_argument("--seed",            type=int, default=42)
parser.add_argument("--limit",           type=int, default=None)
parser.add_argument("--k-values",        default="10,50,100,1000")
args = parser.parse_args()

k_values   = [int(k) for k in args.k_values.split(",")]
bench_dir  = Path(args.benchmark_dir)
cache_path = Path(args.cache) if args.cache else Path("checkpoints/mergesort_100_congress_cache.jsonl")
cache_path.parent.mkdir(parents=True, exist_ok=True)

rng = random.Random(args.seed)

# ── OpenAI client ─────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")

client = AsyncOpenAI(api_key=api_key, timeout=300)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert at matching vague, tip-of-the-tongue descriptions to specific congressional hearing transcript passages.

You will receive a QUERY — a fuzzy, informal description of a moment someone vaguely remembers from a congressional hearing — and CANDIDATE PASSAGES from actual hearing transcripts.

Rank candidates by how well they match the described moment. Focus on:
- The SHAPE of the exchange: who had power, who was being questioned, what rhetorical move was made
- The DYNAMIC: confrontation, evasion, admission, dramatic gesture
- Distinctive details: specific arguments, turning points, emotional beats
- The query may contain WRONG details (wrong year, wrong chamber, conflated memories) — look past surface errors to match the underlying moment

The query will NOT share vocabulary with the correct passage. The person is describing from memory using their own words. You must reason about whether the described dynamic matches what actually happened in each passage.

Every candidate number MUST appear exactly once.

Return ONLY JSON:
{
  "ranked": [
    {"candidate_num": <int>, "reason": "<brief explanation of why this matches or doesn't match the described moment>"},
    ...
  ]
}"""

USER_TEMPLATE = """\
QUERY (someone's fuzzy memory of a hearing moment):
\"\"\"
{query}
\"\"\"

CANDIDATE PASSAGES:
{candidates}

Rank these {k} candidates by how well they match the moment described in the query."""

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

print("[1/4] Loading benchmark...")

corpus  = load_jsonl(bench_dir / "corpus.jsonl")
queries = load_jsonl(bench_dir / "queries.jsonl")

# Qrels
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
print(f"  Qrels: {sum(len(v) for v in qrels.values())} pairs across {len(qrels)} queries")

corpus_ids = list(corpus.keys())
query_ids  = list(queries.keys())

# Query metadata for breakdowns
query_witness = {}
query_memorability = {}
for qid, q in queries.items():
    meta = q.get('metadata', {})
    query_witness[qid] = meta.get('source_speaker', 'unknown').lower()
    query_memorability[qid] = meta.get('memorability', 0)

print(f"  Corpus:  {len(corpus_ids)} passages")
print(f"  Queries: {len(query_ids)}")

# ── Load dense retrieval results ─────────────────────────────────────────────

def load_ranked_lists(path):
    results = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["query_id"])
            results[qid] = [str(x) for x in rec.get("ranked_list", [])]
    return results

print("\n[2/4] Loading dense retrieval results...")

gemini_path  = bench_dir / args.gemini
qwen4b_path  = bench_dir / args.qwen4b
qwen06b_path = bench_dir / args.qwen06b

gemini_results  = load_ranked_lists(gemini_path) if gemini_path.exists() else {}
qwen4b_results  = load_ranked_lists(qwen4b_path) if qwen4b_path.exists() else {}
qwen06b_results = load_ranked_lists(qwen06b_path) if qwen06b_path.exists() else {}

print(f"  Gemini:   {len(gemini_results)} queries (top {args.gemini_top_n})" if gemini_results else "  Gemini:   NOT FOUND")
print(f"  Qwen4B:   {len(qwen4b_results)} queries (top {args.qwen4b_top_n})" if qwen4b_results else "  Qwen4B:   NOT FOUND")
print(f"  Qwen0.6B: {len(qwen06b_results)} queries (top {args.qwen06b_top_n})" if qwen06b_results else "  Qwen0.6B: NOT FOUND")


def get_dense_union(qid):
    def take_top(ranked_list, n):
        return ranked_list[:n]

    gemini_top  = take_top(gemini_results.get(qid,  []), args.gemini_top_n)
    qwen4b_top  = take_top(qwen4b_results.get(qid,  []), args.qwen4b_top_n)
    qwen06b_top = take_top(qwen06b_results.get(qid, []), args.qwen06b_top_n)

    seen  = set()
    union = []
    for cid in gemini_top + qwen4b_top + qwen06b_top:
        if cid not in seen:
            seen.add(cid)
            union.append(cid)

    return union

# ── Listwise sorting ─────────────────────────────────────────────────────────

async def listwise_sort(qid, candidates):
    """Sort candidates by match to the query's described moment."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates[:]

    candidates = [cid for cid in candidates if cid in corpus]
    if not candidates:
        return []

    query_text = queries[qid]['text']
    num_to_cid = {i + 1: cid for i, cid in enumerate(candidates)}

    candidate_lines = "\n\n".join(
        f"[{i+1}]\n{corpus[cid]['text']}"
        for i, cid in enumerate(candidates)
    )

    prompt = USER_TEMPLATE.format(
        query=query_text,
        candidates=candidate_lines,
        k=len(candidates),
    )

    for attempt in range(5):
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
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

            parsed = json.loads(clean.strip())
            ranked = parsed.get("ranked", [])

            out  = []
            seen = set()
            for item in ranked:
                num = item.get("candidate_num")
                if num in num_to_cid and num not in seen:
                    seen.add(num)
                    out.append(num_to_cid[num])

            # Append missing candidates at end
            for cid in candidates:
                if cid not in out:
                    out.append(cid)

            return out

        except KeyboardInterrupt:
            raise
        except Exception as e:
            wait = 10 * (2 ** attempt)
            print(f"\n  [attempt {attempt+1}/5, retry in {wait}s] {e}")
            await asyncio.sleep(wait)

    print(f"\n  [FAILED] Query {qid} - all retries exhausted")
    return None

# ── MergeSortInter algorithm ─────────────────────────────────────────────────

async def mergesortinter(qid, candidates, batch_size=20, top_k=4):
    """
    1. Shuffle candidates
    2. Chunk into batches
    3. Sort each batch in PARALLEL
    4. Take top-k from each batch
    5. Merge-sort top-k together
    6. Interleave remaining candidates round-robin
    """
    if not candidates:
        return []

    candidates = [cid for cid in candidates if cid in corpus]

    if len(candidates) <= batch_size:
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        result = await listwise_sort(qid, shuffled)
        return result if result is not None else []

    # Step 1: Shuffle
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    # Step 2: Chunk
    num_batches = -(-len(shuffled) // batch_size)
    batches = [
        shuffled[i * batch_size : (i + 1) * batch_size]
        for i in range(num_batches)
    ]

    # Step 3: Shuffle each batch, sort in parallel
    for batch in batches:
        rng.shuffle(batch)

    sorted_batches = await asyncio.gather(*[
        listwise_sort(qid, batch) for batch in batches
    ])

    if any(sb is None for sb in sorted_batches):
        print(f"\n  [FAILED] Query {qid} - one or more batches failed")
        return []

    # Step 4: Extract top-k from each batch
    top_candidates = []
    for sorted_batch in sorted_batches:
        top_candidates.extend(sorted_batch[:top_k])

    # Step 5: Shuffle and sort top candidates
    rng.shuffle(top_candidates)
    final_top = await listwise_sort(qid, top_candidates)

    if final_top is None:
        print(f"\n  [FAILED] Query {qid} - final merge sort failed")
        return []

    # Step 6: Interleave remaining
    tails = [sorted_batch[top_k:] for sorted_batch in sorted_batches]

    interleaved = []
    position = 0
    while True:
        added_any = False
        for tail in tails:
            if position < len(tail):
                interleaved.append(tail[position])
                added_any = True
        if not added_any:
            break
        position += 1

    return final_top + interleaved

# ── Load cache ────────────────────────────────────────────────────────────────

cached = {}
if cache_path.exists():
    with open(cache_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("ranked"):
                cached[rec["query_id"]] = rec
    print(f"\n  Loaded {len(cached)} cached results from {cache_path}")

# ── Main loop ─────────────────────────────────────────────────────────────────

print(f"\n[3/4] MergeSortInter reranking with {args.model}...")
print(f"  batch_size={args.batch_size}, top_k={args.top_k}")
print(f"  gemini_top={args.gemini_top_n}, qwen4b_top={args.qwen4b_top_n}, qwen06b_top={args.qwen06b_top_n}")
print(f"  concurrency={args.concurrency} queries at a time")

eval_query_ids = query_ids[:args.limit] if args.limit else query_ids

results    = []
pending    = []
for qid in eval_query_ids:
    if qid in cached:
        results.append(cached[qid])
    else:
        pending.append(qid)
print(f"  Cached:  {len(cached)}")
print(f"  Pending: {len(pending)}")

n_injected_box = [0]
cache_lock = asyncio.Lock()


async def process_one_query(qid, sem, cache_file, pbar):
    async with sem:
        dense_docs   = get_dense_union(qid)
        relevant_ids = list(qrels.get(qid, {}).keys())

        # Gold-inject
        injected = []
        for rel_id in relevant_ids:
            if rel_id not in dense_docs:
                dense_docs.append(rel_id)
                injected.append(rel_id)
        if injected:
            n_injected_box[0] += 1

        # Shuffle before reranking
        rng.shuffle(dense_docs)

        ranked_cids = await mergesortinter(
            qid, dense_docs,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )

        ranked = [{"corpus_id": cid, "rank": i+1} for i, cid in enumerate(ranked_cids)]
        rec = {
            "query_id":   qid,
            "ranked":     ranked,
            "dense_docs": dense_docs,
            "injected":   injected,
        }

        if ranked:
            async with cache_lock:
                cache_file.write(json.dumps(rec) + "\n")
                cache_file.flush()
        else:
            print(f"\n  [SKIPPED CACHE] Query {qid} - empty result")

        pbar.update(1)
        return rec


async def run_all():
    sem = asyncio.Semaphore(args.concurrency)
    cache_file = open(cache_path, "a")
    pbar = tqdm(total=len(pending), desc="MergeSortInter")
    try:
        tasks = [process_one_query(qid, sem, cache_file, pbar) for qid in pending]
        new_recs = await asyncio.gather(*tasks)
    finally:
        pbar.close()
        cache_file.close()
    return new_recs


if pending:
    new_recs = asyncio.run(run_all())
    results.extend(new_recs)

n_injected = n_injected_box[0]

# ── Build run ─────────────────────────────────────────────────────────────────

run = {}
ranked_lists = {}
for rec in results:
    qid    = rec["query_id"]
    ranked = rec.get("ranked", [])
    if not ranked:
        continue
    dense  = rec.get("dense_docs", [])
    docs   = {}
    ranked_set = set()
    total  = len(dense)

    ranked_order = []
    for item in ranked:
        cid  = str(item["corpus_id"])
        rank = item["rank"]
        docs[cid] = float(total - rank + 1)
        ranked_set.add(cid)
        ranked_order.append(cid)

    tail_score = float(len(dense) - len(ranked_set))
    for cid in dense:
        if str(cid) not in ranked_set:
            docs[str(cid)] = tail_score
            tail_score -= 1.0

    run[qid] = docs
    ranked_lists[qid] = ranked_order

# ── Evaluate ──────────────────────────────────────────────────────────────────

print("\n[4/4] Evaluating...")

metrics = set()
for k in k_values:
    metrics.add(f"ndcg_cut_{k}")
    metrics.add(f"recall_{k}")
metrics.add("map")

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
per_query = evaluator.evaluate(run)

ordered = (
    [f"ndcg_cut_{k}" for k in k_values] +
    [f"recall_{k}"   for k in k_values] +
    ["map"]
)

agg = defaultdict(list)
for qid, ms in per_query.items():
    for m, v in ms.items():
        agg[m].append(v)

print(f"\n{'='*60}")
print(f"  GPT reranker — MergeSortInter — Congress ToT")
print(f"  batch_size={args.batch_size}, top_k={args.top_k}")
print(f"  Dense union: Gemini-{args.gemini_top_n} + Qwen4B-{args.qwen4b_top_n} + Qwen0.6B-{args.qwen06b_top_n}")
print(f"  Model: {args.model}")
print(f"{'='*60}")
print(f"  Queries: {len(per_query)}  |  Gold-injected: {n_injected} queries")
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
out_path = bench_dir / f"gpt5_100_mergesortinter{tag_suffix}_results.jsonl"
rows_out = []
for qid in eval_query_ids:
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
        "recall@1000":   round(v.get("recall_1000", 0), 4),
        "map":           round(v.get("map",          0), 4),
        "ranked_list":   ranked_lists.get(qid, []),
    })
out_path.write_text("\n".join(json.dumps(r) for r in rows_out))

summary = {
    "model":           args.model,
    "algorithm":       "mergesortinter",
    "tag":             args.tag,
    "batch_size":      args.batch_size,
    "top_k":           args.top_k,
    "gemini_top_n":    args.gemini_top_n,
    "qwen4b_top_n":    args.qwen4b_top_n,
    "qwen06b_top_n":   args.qwen06b_top_n,
    "concurrency":     args.concurrency,
    "corpus_size":     len(corpus_ids),
    "n_queries":       len(per_query),
    "n_gold_injected": n_injected,
    "metrics":         {m: round(float(np.mean(agg.get(m, [0]))), 4) for m in ordered},
    "per_memorability_ndcg10": {
        str(m): round(float(np.mean(v)), 4) for m, v in sorted(mem_ndcg10.items(), reverse=True)
    },
    "per_witness_ndcg10": {
        w: round(float(np.mean(v)), 4) for w, v in witness_ndcg10.items()
    },
}
summary_path = bench_dir / f"gpt5_100_mergesortinter{tag_suffix}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print(f"\n[+] Per-query results → {out_path}")
print(f"[+] Summary           → {summary_path}")
print(f"[+] Cache             → {cache_path}")
