"""
eval_gpt5_mergesortinter.py
===========================
MergeSortInter GPT reranking for the Writing Analogues benchmark.

Algorithm (mergesortinter):
  1. Union of dense retrieval results + gold injection
  2. Shuffle all candidates
  3. Chunk into ceil(N/batch_size) batches (default 20)
  4. Sort each batch via listwise LLM call (PARALLELIZED)
  5. Extract top-k (default 4) from each batch
  6. Shuffle and sort all top-k candidates together
  7. Interleave remaining candidates round-robin
  8. Evaluate with pytrec_eval

Usage:
    export OPENAI_API_KEY=...
    python eval_gpt5_mergesortinter.py --corpus_dir corpus
    python eval_gpt5_mergesortinter.py --corpus_dir corpus --limit 5
"""

import argparse
import asyncio
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
from openai import AsyncOpenAI
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark-dir",   required=True,
                    help="Directory with corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json")
parser.add_argument("--corpus-dir",      required=True)
parser.add_argument("--gold-track",      default="corpus_track.csv",
                    help="Path to v2 gold CSV for per-author breakdown")
parser.add_argument("--tag",             default="v2",
                    help="Suffix for output files")
parser.add_argument("--model",           default="gpt-5.2")

# Dense retrieval results files (relative to --benchmark-dir)
parser.add_argument("--gemini",          default="")
parser.add_argument("--qwen4b",          default="")
parser.add_argument("--qwen06b",         default="")

parser.add_argument("--gemini-top-n",    type=int, default=155)
parser.add_argument("--qwen4b-top-n",    type=int, default=75)
parser.add_argument("--qwen06b-top-n",   type=int, default=75)

parser.add_argument("--batch-size",      type=int, default=20,
                    help="Candidates per batch for initial listwise sort")
parser.add_argument("--top-k",           type=int, default=4,
                    help="Top-K from each batch to feed into the final merge")
parser.add_argument("--concurrency",     type=int, default=10,
                    help="Max concurrent queries. Each query runs its own batches in parallel internally.")
parser.add_argument("--cache",           default=None,
                    help="Cache file path ")
parser.add_argument("--sleep",           type=float, default=0.3)
parser.add_argument("--seed",            type=int, default=42)
parser.add_argument("--limit",           type=int, default=None)
args = parser.parse_args()

bench_dir  = Path(args.benchmark_dir)
corpus_dir = Path(args.corpus_dir)
cache_path = Path(args.cache) if args.cache else Path("checkpoints/mergesortinter_cache.jsonl")
cache_path.parent.mkdir(parents=True, exist_ok=True)

rng = random.Random(args.seed)

# ── OpenAI client ─────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")

client = AsyncOpenAI(api_key=api_key, timeout=300)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in authorship analysis and stylometry.

You will receive a QUERY SNIPPET and CANDIDATE SNIPPETS. Rank candidates by
how likely they were written by the same author as the query.

Focus on writing style, NOT topic:
- Sentence rhythm and syntactic patterns
- Hedging language and epistemic stance
- Characteristic vocabulary and phrasing
- Tone and rhetorical posture

The same author may write about different topics — topic similarity is NOT
the signal you should rely on.

Every candidate number MUST appear exactly once.

Return ONLY JSON:
{
  "ranked": [
    {"candidate_num": <int>, "reason": "<brief style observation>"},
    ...
  ]
}"""

USER_TEMPLATE = """\
QUERY SNIPPET:
\"\"\"
{query}
\"\"\"

CANDIDATE SNIPPETS:
{candidates}

Rank these {k} candidates by stylistic similarity to the query author."""

# ── Load benchmark ────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d['_id']] = d
    return out

print("[1/4] Loading benchmark...")

corpus  = load_jsonl(bench_dir / "corpus.jsonl")
queries = load_jsonl(bench_dir / "queries.jsonl")

for sid, doc in corpus.items():
    doc['text'] = (corpus_dir / f"{sid}.txt").read_text(encoding="utf-8", errors="replace").strip()
for sid, q in queries.items():
    q['text'] = (corpus_dir / f"{sid}.txt").read_text(encoding="utf-8", errors="replace").strip()

# Binary qrels
qrels_full_path = bench_dir / "qrels.tsv"
qrels = {}
with open(qrels_full_path) as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            qid, cid, score = parts
        elif len(parts) == 4:
            qid, _, cid, score = parts
        else:
            continue
        qrels.setdefault(qid, {})[cid] = int(score)
print(f"  Qrels: {qrels_full_path.name}  ({sum(len(v) for v in qrels.values())} pairs across {len(qrels)} queries)")

with open(bench_dir / "per_query_excluded_ids.json") as f:
    excluded = json.load(f)
excluded = {qid: set(str(x) for x in ids) for qid, ids in excluded.items()}

corpus_ids = list(corpus.keys())
query_ids  = list(queries.keys())

# Author metadata (v2 gold CSV)
snippet_author = {}
csv_path = Path(args.gold_track)
if csv_path.exists():
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [fn.strip() for fn in reader.fieldnames or []]
        for row in reader:
            sid = (row.get("snippet_id") or "").strip()
            author = (row.get("author_name") or "").strip()
            if sid and author:
                snippet_author[sid] = author
    print(f"  Loaded author metadata for {len(snippet_author)} gold snippets")

print(f"  Corpus:  {len(corpus_ids)} snippets")
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
gemini_results  = load_ranked_lists(bench_dir / args.gemini)
qwen4b_results  = load_ranked_lists(bench_dir / args.qwen4b)
qwen06b_results = load_ranked_lists(bench_dir / args.qwen06b)

print(f"  Gemini:   {len(gemini_results)} queries (top {args.gemini_top_n})")
print(f"  Qwen4B:   {len(qwen4b_results)} queries (top {args.qwen4b_top_n})")
print(f"  Qwen0.6B: {len(qwen06b_results)} queries (top {args.qwen06b_top_n})")

def get_dense_union(qid):
    excl = excluded.get(qid, set())

    def filter_and_take(ranked_list, n):
        result = []
        for cid in ranked_list:
            if cid not in excl:
                result.append(cid)
            if len(result) >= n:
                break
        return result

    gemini_top  = filter_and_take(gemini_results.get(qid,  []), args.gemini_top_n)
    qwen4b_top  = filter_and_take(qwen4b_results.get(qid,  []), args.qwen4b_top_n)
    qwen06b_top = filter_and_take(qwen06b_results.get(qid, []), args.qwen06b_top_n)

    seen  = set()
    union = []
    for cid in gemini_top + qwen4b_top + qwen06b_top:
        if cid not in seen:
            seen.add(cid)
            union.append(cid)

    return union

# ── Listwise sorting ─────────────────────────────────────────────────────────

async def listwise_sort(qid, candidates):
    """Sort candidates by authorship similarity. Returns list of corpus_ids in ranked order."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates[:]

    # Filter to candidates in corpus
    candidates = [cid for cid in candidates if cid in corpus]
    if not candidates:
        return []

    query_text = queries[qid]['text']
    num_to_cid = {i + 1: cid for i, cid in enumerate(candidates)}

    candidate_lines = "\n\n".join(
        f"[{i+1}]\n{corpus[cid]['text']}"
        for i, cid in enumerate(candidates)
    )

    system = SYSTEM_PROMPT
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
                    {"role": "system", "content": system},
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

            # Add missing candidates at the end
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

    # All retries failed - return None so we can identify and redo this query
    print(f"\n  [FAILED] Query {qid} - all retries exhausted, returning None")
    return None

# ── MergeSortInter algorithm ─────────────────────────────────────────────────

async def mergesortinter(qid, candidates, batch_size=20, top_k=4):
    """
    MergeSortInter algorithm:
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

    # Step 2: Chunk into batches
    num_batches = -(-len(shuffled) // batch_size)
    batches = [
        shuffled[i * batch_size : (i + 1) * batch_size]
        for i in range(num_batches)
    ]


    # Step 3: Sort each batch IN PARALLEL
    for batch in batches:
        rng.shuffle(batch)

    sorted_batches = await asyncio.gather(*[
        listwise_sort(qid, batch) for batch in batches
    ])

    # Check if any batch failed
    if any(sb is None for sb in sorted_batches):
        print(f"\n  [FAILED] Query {qid} - one or more batches failed")
        return []

    # Step 4: Extract top-k from each batch
    top_candidates = []
    for sorted_batch in sorted_batches:
        top_candidates.extend(sorted_batch[:top_k])


    # Step 5: Shuffle and sort top candidates together
    rng.shuffle(top_candidates)
    final_top = await listwise_sort(qid, top_candidates)

    # Check if final sort failed
    if final_top is None:
        print(f"\n  [FAILED] Query {qid} - final merge sort failed")
        return []


    # Step 6: Interleave remaining candidates
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

    final_ranking = final_top + interleaved

    return final_ranking

# ── Load cache ────────────────────────────────────────────────────────────────

cached = {}
if cache_path.exists():
    with open(cache_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("ranked"):
                cached[rec["query_id"]] = rec
    print(f"\n  Loaded {len(cached)} cached results from {cache_path}")

# ── Main loop (concurrent across queries) ────────────────────────────────────

print(f"\n[3/4] MergeSortInter reranking with {args.model}...")
print(f"  batch_size={args.batch_size}, top_k={args.top_k}")
print(f"  gemini_top={args.gemini_top_n}, qwen4b_top={args.qwen4b_top_n}, qwen06b_top={args.qwen06b_top_n}")
print(f"  concurrency={args.concurrency} queries at a time")

eval_query_ids = query_ids[:args.limit] if args.limit else query_ids

# Separate cached from pending
results    = []
pending    = []
for qid in eval_query_ids:
    if qid in cached:
        results.append(cached[qid])
    else:
        pending.append(qid)
print(f"  Cached:  {len(cached)}")
print(f"  Pending: {len(pending)}")

n_injected_box = [0]  # mutable across coroutines
cache_lock = asyncio.Lock()


async def process_one_query(qid, sem, cache_file, pbar):
    """Build candidate pool, gold-inject, shuffle, mergesortinter, cache."""
    async with sem:
        dense_docs   = get_dense_union(qid)
        relevant_ids = list(qrels.get(qid, {}).keys())

        # Gold-inject: ensure every gold id is in the candidate pool
        injected = []
        for rel_id in relevant_ids:
            if rel_id not in dense_docs and rel_id not in excluded.get(qid, set()):
                dense_docs.append(rel_id)
                injected.append(rel_id)
        if injected:
            n_injected_box[0] += 1

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

        # Only cache successful results (non-empty ranked list)
        if ranked:
            async with cache_lock:
                cache_file.write(json.dumps(rec) + "\n")
                cache_file.flush()
        else:
            print(f"\n  [SKIPPED CACHE] Query {qid} - empty result, will retry on next run")

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
for rec in results:
    qid    = rec["query_id"]
    ranked = rec.get("ranked", [])
    dense  = rec.get("dense_docs", [])
    docs   = {}
    ranked_set = set()
    total  = len(dense)

    for item in ranked:
        cid  = str(item["corpus_id"])
        rank = item["rank"]
        docs[cid] = float(total - rank + 1)
        ranked_set.add(cid)

    # Tail: candidates not in ranked list
    tail_score = float(len(dense) - len(ranked_set))
    for cid in dense:
        if str(cid) not in ranked_set:
            docs[str(cid)] = tail_score
            tail_score -= 1.0

    run[qid] = docs

# ── Evaluate ──────────────────────────────────────────────────────────────────

print("\n[4/4] Evaluating...")

k_values = [10, 50, 100, 1000]
metrics  = set()
for k in k_values:
    metrics.add(f"ndcg_cut_{k}")
    metrics.add(f"recall_{k}")
metrics.add("map")

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
per_query = evaluator.evaluate(run)

def mean_metric(key):
    return np.mean([v.get(key, 0.0) for v in per_query.values()])

ordered = (
    [f"ndcg_cut_{k}" for k in k_values] +
    [f"recall_{k}"   for k in k_values] +
    ["map"]
)

print(f"\n{'='*60}")
print(f"  GPT reranker — mergesortinter")
print(f"  batch_size={args.batch_size}, top_k={args.top_k}")
print(f"  Dense union: Gemini-{args.gemini_top_n} + Qwen4B-{args.qwen4b_top_n} + Qwen0.6B-{args.qwen06b_top_n}")
print(f"  Model: {args.model}")
print(f"{'='*60}")
print(f"  Queries: {len(per_query)}  |  Gold-injected: {n_injected} queries")
print(f"  {'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8}")
print(f"  {'-'*46}")

for metric in ordered:
    vals = [v.get(metric, 0.0) for v in per_query.values()]
    print(f"  {metric:<20} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

if snippet_author:
    print(f"\n  {'─'*56}")
    print(f"  Per-author nDCG@10")
    print(f"  {'─'*56}")
    print(f"  {'Author':<30} {'nDCG@10':>8} {'nDCG@50':>8} {'n':>5}")
    print(f"  {'─'*56}")

    author_ndcg10 = defaultdict(list)
    author_ndcg50 = defaultdict(list)
    for qid, md in per_query.items():
        author = snippet_author.get(qid, "unknown")
        author_ndcg10[author].append(md.get("ndcg_cut_10", 0.0))
        author_ndcg50[author].append(md.get("ndcg_cut_50", 0.0))

    for author in sorted(author_ndcg10, key=lambda a: -np.mean(author_ndcg10[a])):
        v10 = author_ndcg10[author]
        v50 = author_ndcg50[author]
        print(f"  {author:<30} {np.mean(v10):>8.4f} {np.mean(v50):>8.4f} {len(v10):>5}")

print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""
out_path = bench_dir / f"gpt5_mergesortinter{tag_suffix}_results.jsonl"
rows_out = []
for qid in eval_query_ids:
    v = per_query.get(qid, {})
    rows_out.append({
        "query_id":  qid,
        "author":    snippet_author.get(qid, "unknown"),
        "ndcg@10":   round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50":   round(v.get("ndcg_cut_50", 0), 4),
        "recall@10": round(v.get("recall_10",   0), 4),
        "recall@50": round(v.get("recall_50",   0), 4),
        "map":       round(v.get("map",          0), 4),
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
    "n_queries":       len(per_query),
    "n_gold_injected": n_injected,
    "metrics":         {m: round(mean_metric(m), 4) for m in ordered},
    "per_author_ndcg10": {
        author: round(float(np.mean(vals)), 4)
        for author, vals in author_ndcg10.items()
    } if snippet_author else {},
}

summary_path = bench_dir / f"gpt5_mergesortinter{tag_suffix}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"\n[+] Per-query results → {out_path}")
print(f"[+] Summary           → {summary_path}")
print(f"[+] Cache             → {cache_path}")
