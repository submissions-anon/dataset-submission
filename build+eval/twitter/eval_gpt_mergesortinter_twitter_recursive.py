"""
eval_gpt_mergesortinter_twitter_recursive.py
=============================================
MergeSortInter GPT reranking for the Twitter Descriptive-IR benchmark.

RECURSIVE VERSION: Handles large candidate pools (300+) by recursively
applying merge-sort-interleave until the merge pool is small enough for
a direct ranking call.

Algorithm (recursive mergesortinter):
  1. Union candidates from multiple dense retrievers + gold injection
  2. If candidates <= max_final_merge: rank directly (base case)
  3. Otherwise:
     a. Shuffle and chunk into batches of batch_size
     b. Sort each batch via listwise LLM call (parallel)
     c. Extract top-k from each batch for merge pool
     d. Collect tails (positions k+1 onwards)
     e. Recursively process merge pool (go to step 2)
  4. After recursion completes, interleave tails in reverse depth order
     (deepest tails first = highest quality survivors)
  5. Evaluate with pytrec_eval

Requirements:
  pip install pytrec_eval tqdm numpy openai

Usage:
  export OPENAI_API_KEY=...
  python eval_gpt_mergesortinter_twitter_recursive.py --corpus full
  python eval_gpt_mergesortinter_twitter_recursive.py --corpus full --limit 5
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm
from openai import AsyncOpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset")
parser.add_argument("--corpus",      default="full", choices=["implicit", "full"])
parser.add_argument("--model",       default="gpt-5.2")
parser.add_argument("--gemini-results",
                    default="")
parser.add_argument("--qwen-4b-results",
                    default="")
parser.add_argument("--qwen-06b-results",
                    default="")
parser.add_argument("--gemini-top-n",  type=int, default=505)#255
parser.add_argument("--qwen-4b-top-n", type=int, default=250)#125
parser.add_argument("--qwen-06b-top-n",type=int, default=250)#125
parser.add_argument("--batch-size",    type=int, default=20,
                    help="Number of candidates per batch for sorting")
parser.add_argument("--top-k",         type=int, default=4,
                    help="Top-K to extract from each batch for merge pool")
parser.add_argument("--max-final-merge", type=int, default=20,
                    help="Max candidates for direct ranking (triggers recursion if exceeded)")
parser.add_argument("--ckpt",          default="checkpoints/gpt_mergesortinter_twitter_recursive_cache.jsonl")
parser.add_argument("--sleep",         type=float, default=0.3)
parser.add_argument("--seed",          type=int, default=42)
parser.add_argument("--limit",         type=int, default=None)
parser.add_argument("--concurrency",   type=int, default=5,
                    help="Number of queries to process concurrently")
parser.add_argument("--quiet", "-q",   action="store_true",
                    help="Suppress per-query verbose output (recommended for concurrent runs)")
args = parser.parse_args()

rng         = random.Random(args.seed)
dataset_dir = Path(args.dataset_dir)

# ── Client ────────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")
client = AsyncOpenAI(api_key=api_key)

# Concurrency controls (initialized later in async context)
query_semaphore = None
cache_lock = None

# ── Prompts ───────────────────────────────────────────────────────────────────

LISTWISE_SYSTEM = """\
You are an expert in political rhetoric and implicit communication.

You will receive a QUERY asking for tweets that implicitly express a stance,
and a list of CANDIDATE TWEETS. Rank by relevance to the query.

Implicit expression means indirect rhetoric: sarcasm, irony, rhetorical
questions, mockery, selective framing — NOT direct statements.

Ranking:
  - HIGHEST: Clearly expresses the stance through indirect rhetoric
  - MIDDLE: Plausible match but ambiguous
  - LOWER: Direct/explicit statement (not implicit)
  - LOWEST: Does not express the stance

Every candidate number MUST appear exactly once.

Return ONLY JSON:
{
  "ranked": [
    {"candidate_num": <int>, "reason": "<brief>"},
    ...
  ]
}"""

LISTWISE_USER = """\
QUERY: {query}

CANDIDATE TWEETS:
{candidates}

Rank these {k} tweets by relevance to the query."""

# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, tid, score = line.split("\t")
            qrels.setdefault(qid, {})[tid] = int(score)
    return qrels

def load_retriever_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            results[qid] = [(doc_id, score) for doc_id, score in rec.get("ranked", [])]
    return results

def union_candidates(qid, gold_ids, gemini_results, qwen_4b_results, qwen_06b_results):
    seen             = set()
    candidates       = []
    candidate_scores = {}

    def add_from_retriever(retriever_results, top_n, retriever_name):
        if qid not in retriever_results:
            return
        for doc_id, score in retriever_results[qid][:top_n]:
            if doc_id not in seen:
                seen.add(doc_id)
                candidates.append(doc_id)
            key = f"{retriever_name}_score"
            if doc_id not in candidate_scores:
                candidate_scores[doc_id] = {}
            candidate_scores[doc_id][key] = score

    add_from_retriever(gemini_results,   args.gemini_top_n,   "gemini")
    add_from_retriever(qwen_4b_results,  args.qwen_4b_top_n,  "qwen_4b")
    add_from_retriever(qwen_06b_results, args.qwen_06b_top_n, "qwen_06b")

    injected = []
    for tid in gold_ids:
        if tid not in seen:
            seen.add(tid)
            candidates.append(tid)
            injected.append(tid)
            candidate_scores[tid] = {"injected": True}

    return candidates, candidate_scores, injected

# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json_response(raw):
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])
    clean = clean.strip()
    brace = clean.index("{")
    depth, end = 0, brace
    for i, ch in enumerate(clean[brace:], start=brace):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    return json.loads(clean[brace:end+1])

# ── Async listwise sorting ───────────────────────────────────────────────────

async def listwise_sort_async(query_text, candidates):
    """
    Ask LLM to sort a list of tweet candidates asynchronously.
    Returns list of tids in ranked order.
    """
    if not candidates:
        return []

    if len(candidates) == 1:
        return candidates[:]

    # Filter to only candidates in corpus
    candidates = [tid for tid in candidates if tid in corpus_by_id]
    if not candidates:
        return []

    num_to_tid = {i + 1: tid for i, tid in enumerate(candidates)}

    candidate_lines = "\n\n".join(
        f"[{i+1}] {corpus_by_id[tid]['text']}"
        for i, tid in enumerate(candidates)
    )

    prompt = LISTWISE_USER.format(
        query=query_text,
        candidates=candidate_lines,
        k=len(candidates),
    )

    for attempt in range(5):
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": LISTWISE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=1.0,
                max_completion_tokens=128000,
            )
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                raise ValueError("Empty response")

            parsed = parse_json_response(raw)
            ranked = parsed.get("ranked", [])

            # Extract ordered tids
            out  = []
            seen = set()
            for item in ranked:
                num = item.get("candidate_num")
                if num in num_to_tid and num not in seen:
                    seen.add(num)
                    out.append(num_to_tid[num])

            # Add any missing candidates at the end (in original order)
            for tid in candidates:
                if tid not in out:
                    out.append(tid)

            return out

        except KeyboardInterrupt:
            raise
        except Exception as e:
            wait = 10 * (2 ** attempt)
            print(f"\n  [attempt {attempt+1}/5, retry in {wait}s] {e}")
            await asyncio.sleep(wait)

    # Fallback: return original order
    return candidates[:]

# ── Recursive MergeSortInter algorithm ───────────────────────────────────────

async def mergesortinter_recursive(query_text, candidates, depth=0, all_tails=None):
    """
    Recursive merge-sort-interleave reranking.

    Key idea: If the merge pool is still too large after one round of batching,
    recursively apply the same procedure until we get down to max_final_merge.

    Tails are collected at each level and interleaved at the end in reverse depth
    order (deepest tails first, then shallower tails).
    """
    batch_size = args.batch_size
    top_k = args.top_k
    max_final_merge = args.max_final_merge

    if all_tails is None:
        all_tails = []

    indent = "  " * depth

    # Filter to only candidates in corpus
    candidates = [tid for tid in candidates if tid in corpus_by_id]

    # Base case: small enough to rank directly
    if len(candidates) <= max_final_merge:
        if not args.quiet:
            print(f"      {indent}[depth={depth}] Direct ranking of {len(candidates)} candidates")
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        ranked_tids = await listwise_sort_async(query_text, shuffled)
        return ranked_tids, all_tails

    # Shuffle and batch
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    batches = [
        shuffled[i:i + batch_size]
        for i in range(0, len(shuffled), batch_size)
    ]
    n_batches = len(batches)
    if not args.quiet:
        print(f"      {indent}[depth={depth}] {len(candidates)} candidates -> {n_batches} batches of <={batch_size}")

    # Sort each batch in parallel
    tasks = [listwise_sort_async(query_text, batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)

    # Extract top-k from each batch for merge pool, collect tails
    merge_pool = []
    level_tails = []

    for batch_idx, sorted_batch in enumerate(batch_results):
        top_k_docs = sorted_batch[:top_k]
        tail_docs = sorted_batch[top_k:]
        merge_pool.extend(top_k_docs)
        level_tails.append(tail_docs)

    if not args.quiet:
        print(f"      {indent}[depth={depth}] Merge pool: {len(merge_pool)}, tails: {sum(len(t) for t in level_tails)}")

    # Store this level's tails (will be interleaved later, after deeper tails)
    all_tails.append(level_tails)

    # Recursively process merge pool
    rng.shuffle(merge_pool)
    return await mergesortinter_recursive(
        query_text, merge_pool, depth=depth + 1, all_tails=all_tails
    )


async def mergesortinter(query_text, candidates):
    """
    Entry point for recursive merge-sort-interleave.
    Returns list of tids in final ranked order.
    """
    if not args.quiet:
        print(f"      [mergesortinter-recursive] Starting with {len(candidates)} candidates")

    if not candidates:
        return []

    # Get recursive ranking result and all collected tails
    ranked_tids, all_tails = await mergesortinter_recursive(query_text, candidates)

    # Build final ranking
    final_ranking = []
    seen = set()

    # First: add the final ranked results
    for tid in ranked_tids:
        if tid not in seen:
            seen.add(tid)
            final_ranking.append(tid)

    # Then: interleave tails in reverse order (deepest first, then shallower)
    # This ensures higher-quality candidates (from deeper merges) come before
    # lower-quality ones (from earlier/shallower rounds)
    for level_idx, level_tails in reversed(list(enumerate(all_tails))):
        max_tail_len = max(len(t) for t in level_tails) if level_tails else 0
        for pos in range(max_tail_len):
            for batch_tail in level_tails:
                if pos < len(batch_tail):
                    tid = batch_tail[pos]
                    if tid not in seen:
                        seen.add(tid)
                        final_ranking.append(tid)

    if not args.quiet:
        print(f"      [mergesortinter-recursive] Final: {len(final_ranking)} docs ({len(all_tails)} tail levels)")

    return final_ranking


async def process_single_query(qid, query_texts, qrels, gemini_results, qwen_4b_results, qwen_06b_results, cached, cache_file, pbar):
    """Process a single query with semaphore control."""
    global query_semaphore, cache_lock

    async with query_semaphore:
        if qid in cached:
            pbar.update(1)
            return cached[qid], len(cached[qid].get("candidates", [])), False

        query_text = query_texts[qid]
        gold_ids   = list(qrels.get(qid, {}).keys())

        candidates, candidate_scores, injected = union_candidates(
            qid, gold_ids, gemini_results, qwen_4b_results, qwen_06b_results
        )

        # Run recursive mergesortinter
        ranked_tids = await mergesortinter(query_text, candidates)

        # Build ranked list with ranks
        ranked = []
        for rank, tid in enumerate(ranked_tids, start=1):
            ranked.append({
                "tid":  tid,
                "rank": rank,
            })

        rec = {
            "query_id":        qid,
            "ranked":          ranked,
            "candidates":      candidates,
            "injected":        injected,
            "retriever_scores": candidate_scores,
        }

        # Thread-safe cache write
        async with cache_lock:
            cache_file.write(json.dumps(rec) + "\n")
            cache_file.flush()

        pbar.update(1)
        return rec, len(candidates), bool(injected)


async def run_concurrent_reranking(query_ids, query_texts, qrels, gemini_results, qwen_4b_results, qwen_06b_results, cached, ckpt_path):
    """Run reranking for all queries concurrently."""
    global query_semaphore, cache_lock

    query_semaphore = asyncio.Semaphore(args.concurrency)
    cache_lock = asyncio.Lock()

    results_list = []
    n_injected = 0
    union_sizes = []

    cache_file = open(ckpt_path, "a")

    with tqdm(total=len(query_ids), desc=f"MergeSortInter-recursive (concurrency={args.concurrency})") as pbar:
        tasks = [
            process_single_query(qid, query_texts, qrels, gemini_results, qwen_4b_results, qwen_06b_results, cached, cache_file, pbar)
            for qid in query_ids
        ]
        results = await asyncio.gather(*tasks)

    cache_file.close()

    for rec, union_size, was_injected in results:
        results_list.append(rec)
        union_sizes.append(union_size)
        if was_injected:
            n_injected += 1

    return results_list, n_injected, union_sizes

# ── Load everything ───────────────────────────────────────────────────────────

print(f"\n[*] Model      : {args.model}")
print(f"[*] Algorithm  : mergesortinter-RECURSIVE (async, concurrency={args.concurrency})")
print(f"[*] Params     : batch_size={args.batch_size}, top_k={args.top_k}, max_final_merge={args.max_final_merge}")
print(f"[*] Corpus     : {args.corpus}")
print(f"[*] Union      : Gemini top-{args.gemini_top_n} + Qwen4B top-{args.qwen_4b_top_n} + Qwen0.6B top-{args.qwen_06b_top_n}")

print("\n[1/5] Loading corpus...")
corpus_data  = load_jsonl(dataset_dir / f"corpus_{args.corpus}.jsonl")
corpus_by_id = {d["_id"]: d for d in corpus_data}
print(f"  {len(corpus_by_id):,} tweets")

print("\n[2/5] Loading queries & qrels...")
queries_data = load_jsonl(dataset_dir / "queries.jsonl")
qrels        = load_qrels(dataset_dir / "qrels_pooled.tsv")
query_ids    = [q["_id"] for q in queries_data if q["_id"] in qrels]
query_texts  = {q["_id"]: q["text"] for q in queries_data}
if args.limit:
    query_ids = query_ids[:args.limit]
    print(f"  Smoke test: limiting to {args.limit} queries")
print(f"  {len(query_ids)} queries")
avg_rel = np.mean([len(v) for v in qrels.values()])
print(f"  Avg relevant per query: {avg_rel:.1f}")

print("\n[3/5] Loading retriever results...")
gemini_results   = load_retriever_results(args.gemini_results)
qwen_4b_results  = load_retriever_results(args.qwen_4b_results)
qwen_06b_results = load_retriever_results(args.qwen_06b_results)
print(f"  Gemini: {len(gemini_results)} queries")
print(f"  Qwen 4B: {len(qwen_4b_results)} queries")
print(f"  Qwen 0.6B: {len(qwen_06b_results)} queries")

# ── Load cache ────────────────────────────────────────────────────────────────

ckpt_path = Path(args.ckpt)
ckpt_path.parent.mkdir(exist_ok=True)
cached = {}
if ckpt_path.exists():
    with open(ckpt_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("ranked"):
                cached[rec["query_id"]] = rec
    print(f"\n[4/5] Loaded {len(cached)} cached results from {ckpt_path}")

# ── Reranking loop (concurrent) ──────────────────────────────────────────────

print(f"\n[5/5] MergeSortInter-RECURSIVE reranking with {args.model} (concurrency={args.concurrency})...")

results_list, n_injected, union_sizes = asyncio.run(
    run_concurrent_reranking(
        query_ids, query_texts, qrels,
        gemini_results, qwen_4b_results, qwen_06b_results,
        cached, ckpt_path
    )
)

# ── Build run dict ────────────────────────────────────────────────────────────

run = {}
for rec in results_list:
    qid        = rec["query_id"]
    ranked     = rec.get("ranked", [])
    candidates = rec.get("candidates", [])

    docs       = {}
    ranked_set = set()
    total      = len(candidates)

    for item in ranked:
        tid  = item["tid"]
        rank = item["rank"]
        docs[tid] = float(total - rank + 1)
        ranked_set.add(tid)

    # Any candidates not in ranked list get lowest scores
    tail_score = float(total - len(ranked_set))
    for tid in candidates:
        if tid not in ranked_set:
            docs[tid] = tail_score
            tail_score -= 1.0

    run[qid] = docs

# ── Evaluate ──────────────────────────────────────────────────────────────────

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {
        "ndcg_cut.10,50,100",
        "recall.10,50,100,300,500",
        "recip_rank",
        "success.1,5,10",
    }
)
results = evaluator.evaluate(run)

def mean(key):
    return float(np.mean([v.get(key, 0.0) for v in results.values()]))

best_ranks = []
for qid in query_ids:
    rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    if ranks:
        best_ranks.append(min(ranks))
best_ranks_arr = np.array(best_ranks) if best_ranks else np.array([])

avg_union = np.mean(union_sizes) if union_sizes else 0
print(f"\n{'='*75}")
print(f"  {args.model}  —  mergesortinter-RECURSIVE")
print(f"  batch_size={args.batch_size}, top_k={args.top_k}, max_final_merge={args.max_final_merge}")
print(f"  Union(Gemini-{args.gemini_top_n}+Qwen4B-{args.qwen_4b_top_n}+Qwen0.6B-{args.qwen_06b_top_n}) + gold-inject")
print(f"  Corpus: {args.corpus} ({len(corpus_by_id):,} tweets)  |  Queries: {len(query_ids)}")
print(f"  Avg union size: {avg_union:.1f}  |  Gold-injected: {n_injected} queries")
print(f"{'='*75}")
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
print(f"  {'Recall@300':<22} {mean('recall_300'):>8.4f}")
print(f"  {'Recall@500':<22} {mean('recall_500'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Success@1':<22} {mean('success_1'):>8.4f}")
print(f"  {'Success@5':<22} {mean('success_5'):>8.4f}")
print(f"  {'Success@10':<22} {mean('success_10'):>8.4f}")
if len(best_ranks_arr):
    print(f"  {'-'*32}")
    print(f"  Best-relevant-doc rank distribution:")
    for thresh in [1, 5, 10, 50, 100, 300, 500]:
        n = int((best_ranks_arr <= thresh).sum())
        print(f"    Top-{thresh:<5} {n:>4} / {len(query_ids)}  ({n/len(query_ids):.1%})")
print(f"{'='*75}")

# ── Per-query breakdown ───────────────────────────────────────────────────────

print(f"\n  {'qid':<8} {'n_rel':>5}  {'NDCG@10':>8}  {'R@10':>7}  {'R@50':>7}  {'R@100':>7}  {'R@300':>7}  {'R@500':>7}  {'best_rank':>9}")
print(f"  {'-'*85}")
for qid in query_ids:
    v        = results.get(qid, {})
    rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    best     = min(ranks) if ranks else -1
    print(f"  {qid:<8} {len(qrels[qid]):>5}  "
          f"{v.get('ndcg_cut_10',0):>8.4f}  "
          f"{v.get('recall_10',0):>7.4f}  "
          f"{v.get('recall_50',0):>7.4f}  "
          f"{v.get('recall_100',0):>7.4f}  "
          f"{v.get('recall_300',0):>7.4f}  "
          f"{v.get('recall_500',0):>7.4f}  "
          f"{best:>9}")

# ── Save ──────────────────────────────────────────────────────────────────────

model_slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
out_path   = dataset_dir / f"results_gpt_mergesortinter_recursive_{model_slug}_{args.corpus}.jsonl"
rows = []
for qid in query_ids:
    v        = results.get(qid, {})
    rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    rows.append({
        "query_id":     qid,
        "best_rank":    min(ranks) if ranks else None,
        "n_relevant":   len(qrels[qid]),
        "mrr":          round(v.get("recip_rank",   0), 4),
        "ndcg@10":      round(v.get("ndcg_cut_10",  0), 4),
        "ndcg@50":      round(v.get("ndcg_cut_50",  0), 4),
        "ndcg@100":     round(v.get("ndcg_cut_100", 0), 4),
        "recall@10":    round(v.get("recall_10",    0), 4),
        "recall@50":    round(v.get("recall_50",    0), 4),
        "recall@100":   round(v.get("recall_100",   0), 4),
        "recall@300":   round(v.get("recall_300",   0), 4),
        "recall@500":   round(v.get("recall_500",   0), 4),
        "success@1":    round(v.get("success_1",    0), 4),
        "success@5":    round(v.get("success_5",    0), 4),
        "success@10":   round(v.get("success_10",   0), 4),
        "ranked":       ranked[:100],
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Per-query results -> {out_path}")
print(f"[+] Cache             -> {ckpt_path}  (safe to resume from)")
