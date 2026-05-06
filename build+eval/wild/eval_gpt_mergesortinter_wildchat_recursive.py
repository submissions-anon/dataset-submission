"""
eval_gpt_mergesortinter_wildchat_recursive.py
==============================================
MergeSortInter GPT reranking for the WildChat Descriptive-IR benchmark.

RECURSIVE VERSION: Handles large candidate pools (300+) by recursively
applying merge-sort-interleave until the merge pool is small enough for
a direct ranking call.

Algorithm (recursive mergesortinter):
  1. Load top-N from Gemini, top-M from each Qwen (0.6B and 4B) + gold injection
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
  python eval_gpt_mergesortinter_wildchat_recursive.py --dataset-dir dataset/merged
  python eval_gpt_mergesortinter_wildchat_recursive.py --dataset-dir dataset/merged --limit 5
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir",   default="dataset/abstract")
parser.add_argument("--model",         default="gpt-5.2")
parser.add_argument("--gemini-top-n",  type=int, default=155)#55
parser.add_argument("--qwen-top-n",    type=int, default=75) #25
parser.add_argument("--batch-size",    type=int, default=20,
                    help="Number of candidates per batch for sorting") 
parser.add_argument("--top-k",         type=int, default=4,
                    help="Top-K to extract from each batch for merge pool")
parser.add_argument("--max-final-merge", type=int, default=20,
                    help="Max candidates for direct ranking (triggers recursion if exceeded)")
parser.add_argument("--ckpt",          default="checkpoints/gpt_mergesortinter_wildchat_abstract_recursive_cache.jsonl")
parser.add_argument("--sleep",         type=float, default=0.3)
parser.add_argument("--seed",          type=int, default=42)
parser.add_argument("--limit",         type=int, default=None)
parser.add_argument("--max-doc-chars", type=int, default=999999)
parser.add_argument("--concurrency",   type=int, default=10, help="Max concurrent query processing")
args = parser.parse_args()

rng         = random.Random(args.seed)
dataset_dir = Path(args.dataset_dir)

corpus_file   = Path("dataset") / "corpus.jsonl"
queries_file  = dataset_dir / "queries.jsonl"
qrels_file    = dataset_dir / "qrels.tsv"
gemini_file   = dataset_dir / ""
qwen_06b_file = dataset_dir / ""
qwen_4b_file  = dataset_dir / ""

for f in [corpus_file, queries_file, qrels_file, gemini_file, qwen_06b_file, qwen_4b_file]:
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

# ── Client ────────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")
client = AsyncOpenAI(api_key=api_key)

# ── Prompts ───────────────────────────────────────────────────────────────────

LISTWISE_SYSTEM = """\
You are an expert at identifying LLM failure modes in human-AI conversations.

You will receive a QUERY describing an abstract LLM failure pattern, and a
numbered list of CANDIDATE CONVERSATIONS.

Your task is to rank the candidates from most to least relevant to the query.
A candidate is relevant if the AI assistant in that conversation exhibits the
failure pattern described in the query.

What matters is the TYPE OF MISTAKE, not the subject matter. Rank by how
precisely and clearly each candidate exhibits the failure pattern.

Ranking guidelines:
  - A candidate is relevant if the user's instruction matches the constraint
    type in the query AND the AI violates it in the way the query describes.
  - Candidates that clearly exhibit the failure pattern should rank higher.
  - Candidates where the failure is uncertain or partial should rank lower.
  - Candidates that don't match the failure pattern should rank lowest.

Every candidate number MUST appear exactly once in your output.

Return ONLY a JSON object:
{
  "ranked": [
    {"candidate_num": <int>, "reason": "<brief phrase explaining relevance>"},
    ...
  ]
}"""

LISTWISE_USER = """\
QUERY (failure pattern to find):
{query}

CANDIDATE CONVERSATIONS:
{candidates}

Rank these {k} candidates from most to least relevant to the failure pattern."""

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
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, did, score = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels

def load_dense_results(path):
    results = {}
    for rec in load_jsonl(path):
        qid  = rec["query_id"]
        hits = rec.get("results", rec.get("hits", rec.get("ranked", [])))
        ranked = []
        for h in hits:
            if isinstance(h, dict):
                ranked.append((h.get("doc_id", h.get("_id")), h.get("score", 0.0)))
            else:
                ranked.append((h[0], h[1] if len(h) > 1 else 0.0))
        results[qid] = ranked
    return results

def get_union_candidates(qid, gemini_results, qwen_06b_results, qwen_4b_results,
                         gemini_top_n, qwen_top_n, gold_ids=None):
    candidates      = []
    candidate_scores = {}
    seen            = set()

    for did, score in gemini_results.get(qid, [])[:gemini_top_n]:
        if did not in seen:
            candidates.append(did)
            seen.add(did)
            candidate_scores[did] = float(score)

    for did, score in qwen_06b_results.get(qid, [])[:qwen_top_n]:
        if did not in seen:
            candidates.append(did)
            seen.add(did)
            candidate_scores[did] = float(score)

    for did, score in qwen_4b_results.get(qid, [])[:qwen_top_n]:
        if did not in seen:
            candidates.append(did)
            seen.add(did)
            candidate_scores[did] = float(score)

    injected = []
    if gold_ids:
        for did in gold_ids:
            if did not in seen:
                candidates.append(did)
                seen.add(did)
                candidate_scores[did] = 0.0
                injected.append(did)

    return candidates, candidate_scores, injected

# ── Async listwise sorting ───────────────────────────────────────────────────

async def listwise_sort_async(query_text, candidates):
    """
    Ask LLM to sort a list of candidates asynchronously.
    Returns list of dids in ranked order.
    """
    if not candidates:
        return []

    if len(candidates) == 1:
        return candidates[:]

    num_to_did = {i + 1: did for i, did in enumerate(candidates)}

    candidate_lines = "\n\n".join(
        f"[{i+1}] {corpus_by_id[did]['text'][:args.max_doc_chars]}"
        f"{'...' if len(corpus_by_id[did]['text']) > args.max_doc_chars else ''}"
        for i, did in enumerate(candidates)
        if did in corpus_by_id
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
            parsed = json.loads(clean[brace:end+1])
            ranked = parsed.get("ranked", [])

            # Extract ordered dids
            out  = []
            seen = set()
            for item in ranked:
                num = item.get("candidate_num")
                if num in num_to_did and num not in seen:
                    seen.add(num)
                    out.append(num_to_did[num])

            # Add any missing candidates at the end (in original order)
            for did in candidates:
                if did not in out:
                    out.append(did)

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

    # Base case: small enough to rank directly
    if len(candidates) <= max_final_merge:
        print(f"      {indent}[depth={depth}] Direct ranking of {len(candidates)} candidates")
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        ranked_dids = await listwise_sort_async(query_text, shuffled)
        return ranked_dids, all_tails

    # Shuffle and batch
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    batches = [
        shuffled[i:i + batch_size]
        for i in range(0, len(shuffled), batch_size)
    ]
    n_batches = len(batches)
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
    Returns list of dids in final ranked order.
    """
    print(f"      [mergesortinter-recursive] Starting with {len(candidates)} candidates")

    if not candidates:
        return []

    # Get recursive ranking result and all collected tails
    ranked_dids, all_tails = await mergesortinter_recursive(query_text, candidates)

    # Build final ranking
    final_ranking = []
    seen = set()

    # First: add the final ranked results
    for did in ranked_dids:
        if did not in seen:
            seen.add(did)
            final_ranking.append(did)

    # Then: interleave tails in reverse order (deepest first, then shallower)
    # This ensures higher-quality candidates (from deeper merges) come before
    # lower-quality ones (from earlier/shallower rounds)
    for level_idx, level_tails in reversed(list(enumerate(all_tails))):
        max_tail_len = max(len(t) for t in level_tails) if level_tails else 0
        for pos in range(max_tail_len):
            for batch_tail in level_tails:
                if pos < len(batch_tail):
                    did = batch_tail[pos]
                    if did not in seen:
                        seen.add(did)
                        final_ranking.append(did)

    print(f"      [mergesortinter-recursive] Final: {len(final_ranking)} docs ({len(all_tails)} tail levels)")

    return final_ranking


async def process_query(semaphore, qid):
    """Process a single query with recursive mergesortinter (async)."""
    async with semaphore:
        gold_ids = list(qrels.get(qid, {}).keys())

        candidates, candidate_scores, injected = get_union_candidates(
            qid, gemini_results, qwen_06b_results, qwen_4b_results,
            args.gemini_top_n, args.qwen_top_n, gold_ids
        )

        print(f"\n  [{qid}] {len(candidates)} candidates")

        # Run recursive mergesortinter
        ranked_dids = await mergesortinter(query_texts[qid], candidates)

        # Build ranked list
        ranked = []
        for rank, did in enumerate(ranked_dids, start=1):
            ranked.append({"did": did, "rank": rank})

        return {
            "query_id": qid,
            "ranked": ranked,
            "candidates": candidates,
            "injected": injected,
            "dense_scores": candidate_scores,
        }

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"\n[*] Model      : {args.model}")
print(f"[*] Algorithm  : mergesortinter-RECURSIVE")
print(f"[*] Params     : batch_size={args.batch_size}, top_k={args.top_k}, max_final_merge={args.max_final_merge}")
print(f"[*] Gemini top-N: {args.gemini_top_n}  |  Qwen top-N: {args.qwen_top_n}")
print(f"[*] Dataset    : {dataset_dir}")

print("\n[1/4] Loading corpus...")
corpus_by_id = {}
for doc in tqdm(iter_corpus(corpus_file), desc="  Loading", unit="doc"):
    corpus_by_id[doc["_id"]] = {"text": doc["text"]}
print(f"  {len(corpus_by_id):,} documents")

print("\n[2/4] Loading queries & qrels...")
queries   = load_queries(queries_file)
qrels     = load_qrels(qrels_file)
query_ids = [qid for qid in queries if qid in qrels]
query_texts = {qid: queries[qid] for qid in query_ids}

if args.limit:
    query_ids = query_ids[:args.limit]
    print(f"  Smoke test: limiting to {args.limit} queries")

print(f"  {len(query_ids)} queries with qrels")
counts  = [len(v) for v in qrels.values()]
avg_rel = float(np.mean(counts))
print(f"  Avg rel/query: {avg_rel:.1f}  |  Min/Max: {min(counts)}/{max(counts)}")

print("\n[3/4] Loading dense retrieval results...")
gemini_results   = load_dense_results(gemini_file)
qwen_06b_results = load_dense_results(qwen_06b_file)
qwen_4b_results  = load_dense_results(qwen_4b_file)
print(f"  Loaded {len(gemini_results)} / {len(qwen_06b_results)} / {len(qwen_4b_results)} queries")

# ── Load checkpoint ───────────────────────────────────────────────────────────

ckpt_path = Path(args.ckpt)
ckpt_path.parent.mkdir(exist_ok=True)
cached = {}
if ckpt_path.exists():
    with open(ckpt_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("ranked"):
                cached[rec["query_id"]] = rec
    print(f"\n  Loaded {len(cached)} cached results from {ckpt_path}")

# ── Async reranking ──────────────────────────────────────────────────────────

async def run_reranking():
    """Run reranking for all queries concurrently."""
    print(f"\n[4/4] MergeSortInter-RECURSIVE reranking with {args.model} (concurrency={args.concurrency})...")

    results_list = []
    to_process = []

    # Separate cached vs new
    for qid in query_ids:
        if qid in cached:
            results_list.append(cached[qid])
        else:
            to_process.append(qid)

    print(f"  Cached: {len(results_list)}, To process: {len(to_process)}")

    if to_process:
        semaphore = asyncio.Semaphore(args.concurrency)

        # Process queries concurrently
        new_results = await tqdm_asyncio.gather(
            *[process_query(semaphore, qid) for qid in to_process],
            desc="Processing queries"
        )

        # Save to cache
        cache_file = open(ckpt_path, "a")
        for rec in new_results:
            results_list.append(rec)
            cache_file.write(json.dumps(rec) + "\n")
        cache_file.close()

    # Compute stats
    total_candidates = [len(r.get("candidates", [])) for r in results_list]
    n_injected = sum(1 for r in results_list if r.get("injected"))

    if total_candidates:
        print(f"\n  Avg candidates per query: {np.mean(total_candidates):.1f}")
        print(f"  Queries with gold injection: {n_injected}")

    return results_list, n_injected


# Run async reranking
results_list, n_injected = asyncio.run(run_reranking())

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
        did  = item["did"]
        rank = item["rank"]
        docs[did] = float(total - rank + 1)
        ranked_set.add(did)

    # Any candidates not in ranked list get lowest scores
    tail_score = float(total - len(ranked_set))
    for did in candidates:
        if did not in ranked_set:
            docs[did] = tail_score
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
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    if ranks:
        best_ranks.append(min(ranks))
best_ranks_arr = np.array(best_ranks) if best_ranks else np.array([])

print(f"\n{'='*65}")
print(f"  {args.model}  —  mergesortinter-RECURSIVE")
print(f"  batch_size={args.batch_size}, top_k={args.top_k}, max_final_merge={args.max_final_merge}")
print(f"  Gemini(top-{args.gemini_top_n}) + Qwens(top-{args.qwen_top_n} each) + gold-inject")
print(f"  Corpus: {len(corpus_by_id):,} docs  |  Queries: {len(query_ids)}")
print(f"  Gold-injected: {n_injected}  |  Avg relevant: {avg_rel:.1f}")
print(f"{'='*65}")
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
    print(f"  Best-relevant-doc rank:")
    for thresh in [1, 5, 10, 50, 100, 300, 500]:
        n = int((best_ranks_arr <= thresh).sum())
        print(f"    Top-{thresh:<5} {n:>3}/{len(query_ids)}  ({n/len(query_ids):.0%})")
print(f"{'='*65}")

# ── Per-query breakdown ───────────────────────────────────────────────────────

print(f"\n  {'qid':<8} {'n_rel':>5}  {'NDCG@10':>8}  {'R@10':>7}  {'R@50':>7}  {'R@100':>7}  {'R@300':>7}  {'R@500':>7}  {'best_rank':>9}")
print(f"  {'-'*81}")
for qid in query_ids:
    v        = results.get(qid, {})
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
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
out_path   = dataset_dir / f"results_gpt_mergesortinter_abstract_300_recursive_{model_slug}.jsonl"

rows = []
for qid in query_ids:
    v        = results.get(qid, {})
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    rec      = next((r for r in results_list if r["query_id"] == qid), {})
    rows.append({
        "query_id":    qid,
        "query_text":  queries[qid][:120],
        "best_rank":   min(ranks) if ranks else None,
        "n_relevant":  len(rel_docs),
        "mrr":         round(v.get("recip_rank",   0), 4),
        "ndcg@10":     round(v.get("ndcg_cut_10",  0), 4),
        "ndcg@50":     round(v.get("ndcg_cut_50",  0), 4),
        "ndcg@100":    round(v.get("ndcg_cut_100", 0), 4),
        "recall@10":   round(v.get("recall_10",    0), 4),
        "recall@50":   round(v.get("recall_50",    0), 4),
        "recall@100":  round(v.get("recall_100",   0), 4),
        "recall@300":  round(v.get("recall_300",   0), 4),
        "recall@500":  round(v.get("recall_500",   0), 4),
        "success@1":   round(v.get("success_1",    0), 4),
        "success@5":   round(v.get("success_5",    0), 4),
        "success@10":  round(v.get("success_10",   0), 4),
        "top10":       ranked,
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Per-query results → {out_path}")
print(f"[+] Cache             → {ckpt_path}  (safe to resume from)")
