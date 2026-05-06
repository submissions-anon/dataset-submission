"""
eval_gpt_mergesortinter_with_solutions.py
==========================================
GPT-5 reranking with merge-sort-interleave for Math Reasoning-Analogue benchmark.

KEY DIFFERENCE FROM ORIGINAL: Includes full solutions in the prompt so GPT can
compare actual reasoning styles/meta-moves rather than inferring them.

Pipeline:
  1. Dense retrieval candidates: union of Gemini + Qwen embeddings
  2. Gold-inject: all qrel docs guaranteed in the candidate pool
  3. GPT reranks via merge-sort-interleave with SOLUTIONS VISIBLE
  4. Evaluate with pytrec_eval

CONCURRENCY: Processes multiple queries in parallel for faster execution.
Use --concurrency to control parallelism (default: 8).

Usage:
  export OPENAI_API_KEY=...
  python eval_gpt_mergesortinter_with_solutions.py
  python eval_gpt_mergesortinter_with_solutions.py --concurrency 16  # more parallel queries
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
from openai import AsyncOpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir",  default="dataset")
parser.add_argument("--model",        default="gpt-5.2")
parser.add_argument("--gemini",       default="")
parser.add_argument("--qwen4b",       default="")
parser.add_argument("--qwen06b",      default="")
parser.add_argument("--gemini-top",   type=int, default=265)#165 #65
parser.add_argument("--qwen4b-top",   type=int, default=135) #85 #35
parser.add_argument("--qwen06b-top",  type=int, default=135) #85 #35
parser.add_argument("--batch-size",   type=int, default=4,
                    help="Candidates per batch (smaller due to longer context with solutions)")
parser.add_argument("--batch-top-k",  type=int, default=2,
                    help="Top-k from each batch for merge pool")
parser.add_argument("--max-final-merge", type=int, default=16,
                    help="Max candidates for final direct ranking (triggers recursion if exceeded)")
parser.add_argument("--ckpt",         default="checkpoints/gpt_mergesortinter_with_solutions_500.jsonl")
parser.add_argument("--sleep",        type=float, default=0.1)
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--limit",        type=int, default=None)
parser.add_argument("--solutions-file", default="final_dataset.json",
                    help="Path to JSON file with problem solutions")
parser.add_argument("--concurrency",  type=int, default=8,
                    help="Number of queries to process concurrently")
args = parser.parse_args()

rng         = random.Random(args.seed)
dataset_dir = Path(args.dataset_dir)

# ── Client ────────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")
client = AsyncOpenAI(api_key=api_key)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert mathematician. Your task is to identify problems that share
the same ABSTRACT PROOF STRATEGY — the same "meta-move" or reasoning pattern.

You are given BOTH the problem statements AND their solutions. Use the solutions
to identify the core reasoning technique, not just surface-level topic similarity.

WHAT "SAME ABSTRACT STRATEGY" MEANS:

Two problems match if their solutions use the same high-level meta-move. Examples
(though there could be many more) are:
  - "Transform representation → swap order of operations → collapse"
  - "Symmetry averaging to project onto invariants"
  - "Sign-reversing involution for cancellation"
  - "Linearize near fixed point for asymptotics"

Problems can share a meta-move despite having NOTHING in common on the surface:
  - A number theory problem and an analysis problem might both use
    "uniqueness of a canonical object" as the key insight
  - An integral evaluation and a combinatorial sum might both use
    "swap order of operations then collapse"

IGNORE surface similarity:
  ✗ Same mathematical field (analysis, algebra, geometry)
  ✗ Same objects (integrals, groups, matrices)
  ✗ Similar notation or formulas
  ✗ Problems that "look alike"

FOCUS ON the deep proof skeleton revealed in the solutions:
  ✓ What is the KEY INSIGHT that unlocks each problem?
  ✓ What meta-level trick or principle does each solution rely on?
  ✓ Would the same proof OUTLINE work for both problems?
  ✓ Do the solutions follow the same structural pattern?

Rank ALL {k} candidates. Return ONLY JSON:
{{
  "ranked": [
    {{"candidate_num": <int>, "reason": "<shared meta-move, 5-15 words>"}},
    ...
  ]
}}"""

USER_TEMPLATE = """\
QUERY PROBLEM AND SOLUTION:
═══════════════════════════
PROBLEM:
{query_problem}

SOLUTION:
{query_solution}

═══════════════════════════════════════════════════════════════════════════════

CANDIDATE PROBLEMS AND SOLUTIONS:
{candidates}

═══════════════════════════════════════════════════════════════════════════════

INSTRUCTIONS:
1. Analyze the query solution: What is the core proof technique/meta-move?
2. For each candidate, analyze its solution for the same: What's its core technique?
3. Rank by whether the ABSTRACT PROOF STRATEGY matches — not topic similarity.

Key question: Could you describe both solutions with the same high-level template?
For example: "Both use a sign-reversing involution to cancel terms pairwise."
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

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
            qid, did, score = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels

def load_dense_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            ranked = rec.get("ranked", [])
            results[qid] = [(did, score) for did, score in ranked]
    return results

def load_solutions(path):
    """Load solutions from final_dataset.json, keyed by problem_id."""
    solutions = {}
    with open(path) as f:
        data = json.load(f)
    for item in data:
        # Build key matching corpus _id format: competition___year___problem_id
        key = f"{item['competition']}___{item['year']}___{item['problem_id']}"
        solutions[key] = item.get("solution", "")
    return solutions

def truncate_solution(solution, max_chars=999999):
    """Truncate solution to avoid context overflow while keeping key parts."""
    if len(solution) <= max_chars:
        return solution
    # Keep first part (main solution) and truncate
    return solution[:max_chars] + "\n\n[... solution truncated for length ...]"

def get_union_candidates(qid, dense_results_list, top_k_list, gold_ids, excluded_ids=None):
    """Build candidate pool from dense retrievers, excluding query duplicates."""
    excluded_ids = excluded_ids or set()
    candidate_scores = {}
    seen = set()
    candidates = []

    for dense_results, top_k in zip(dense_results_list, top_k_list):
        if qid not in dense_results:
            continue
        ranked = dense_results[qid][:top_k]
        for did, score in ranked:
            if did in seen or did in set(gold_ids) or did in excluded_ids:
                continue
            seen.add(did)
            candidates.append(did)
            candidate_scores[did] = max(candidate_scores.get(did, 0.0), float(score))

    injected = []
    for did in gold_ids:
        if did not in seen and did not in excluded_ids:
            candidates.append(did)
            injected.append(did)
            seen.add(did)
            candidate_scores[did] = candidate_scores.get(did, 0.0)

    return candidates, candidate_scores, injected

# ── GPT reranking ─────────────────────────────────────────────────────────────

async def gpt_rerank_batch_async(query_problem_text, query_solution_text,
                                  candidates, corpus_by_id, solutions_by_id):
    """Rerank a single batch of candidates asynchronously, WITH SOLUTIONS."""
    num_to_did = {i + 1: did for i, did in enumerate(candidates)}

    def format_candidate(i, did):
        if did not in corpus_by_id:
            return None
        problem_text = corpus_by_id[did]['text']
        solution_text = solutions_by_id.get(did, "[Solution not available]")
        solution_text = truncate_solution(solution_text)
        return f"""[{i+1}] ─────────────────────────────────────────
PROBLEM:
{problem_text}

SOLUTION:
{solution_text}
"""

    candidate_lines = "\n".join(
        line for line in (format_candidate(i, did) for i, did in enumerate(candidates))
        if line is not None
    )

    system = SYSTEM_PROMPT.format(k=len(candidates))
    prompt = USER_TEMPLATE.format(
        query_problem=query_problem_text,
        query_solution=truncate_solution(query_solution_text),
        candidates=candidate_lines,
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

            out = []
            seen = set()
            for item in ranked:
                num = item.get("candidate_num")
                if num in num_to_did and num not in seen:
                    seen.add(num)
                    out.append({
                        "did":    num_to_did[num],
                        "rank":   len(out) + 1,
                        "reason": item.get("reason", ""),
                    })
            return out

        except KeyboardInterrupt:
            raise
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"\n  [batch attempt {attempt+1}/5, retry in {wait}s] {e}")
            await asyncio.sleep(wait)
    return []


async def gpt_mergesortinter_recursive(query_problem_text, query_solution_text,
                                        candidates, corpus_by_id, solutions_by_id,
                                        depth=0, all_tails=None):
    """
    Recursive merge-sort-interleave reranking with solutions visible.

    Key idea: If the merge pool is still too large after one round of batching,
    recursively apply the same procedure until we get down to a manageable size.

    Tails are collected at each level and interleaved at the end in reverse depth order
    (deepest tails first, then shallower tails).
    """
    batch_size = args.batch_size
    batch_top_k = args.batch_top_k
    max_final_merge = args.max_final_merge

    if all_tails is None:
        all_tails = []

    indent = "  " * depth

    # Base case: small enough to rank directly
    if len(candidates) <= max_final_merge:
        print(f"    {indent}[depth={depth}] Direct ranking of {len(candidates)} candidates")
        result = await gpt_rerank_batch_async(
            query_problem_text, query_solution_text, candidates, corpus_by_id, solutions_by_id
        )
        return result, all_tails

    # Shuffle and batch
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    batches = [
        shuffled[i:i + batch_size]
        for i in range(0, len(shuffled), batch_size)
    ]
    n_batches = len(batches)
    print(f"    {indent}[depth={depth}] {len(candidates)} candidates → {n_batches} batches of ≤{batch_size}")

    # Sort each batch in parallel
    tasks = [
        gpt_rerank_batch_async(
            query_problem_text, query_solution_text, batch, corpus_by_id, solutions_by_id
        )
        for batch in batches
    ]
    batch_results = await asyncio.gather(*tasks)

    # Extract top-k from each batch for merge pool, collect tails
    merge_pool = []
    level_tails = []

    for batch_idx, result in enumerate(batch_results):
        top_k_docs = [item["did"] for item in result[:batch_top_k]]
        tail_docs = [item["did"] for item in result[batch_top_k:]]
        merge_pool.extend(top_k_docs)
        level_tails.append(tail_docs)

    print(f"    {indent}[depth={depth}] Merge pool: {len(merge_pool)}, tails: {sum(len(t) for t in level_tails)}")

    # Store this level's tails (will be interleaved later, after deeper tails)
    all_tails.append(level_tails)

    # Recursively process merge pool
    rng.shuffle(merge_pool)
    return await gpt_mergesortinter_recursive(
        query_problem_text, query_solution_text,
        merge_pool, corpus_by_id, solutions_by_id,
        depth=depth + 1, all_tails=all_tails
    )


async def gpt_mergesortinter(query_problem_text, query_solution_text,
                              candidates, corpus_by_id, solutions_by_id):
    """
    Entry point for recursive merge-sort-interleave.
    """
    print(f"    [mergesortinter+sol] Starting with {len(candidates)} candidates")

    # Get recursive ranking result and all collected tails
    ranked_result, all_tails = await gpt_mergesortinter_recursive(
        query_problem_text, query_solution_text,
        candidates, corpus_by_id, solutions_by_id
    )

    # Build final ranking
    final_ranked = []
    seen = set()

    # First: add the final ranked results
    for item in ranked_result:
        if item["did"] not in seen:
            seen.add(item["did"])
            final_ranked.append({
                "did":    item["did"],
                "rank":   len(final_ranked) + 1,
                "reason": item["reason"],
            })

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
                        final_ranked.append({
                            "did":    did,
                            "rank":   len(final_ranked) + 1,
                            "reason": f"tail-depth-{level_idx}",
                        })

    print(f"    [mergesortinter+sol] Final ranking: {len(final_ranked)} docs ({len(all_tails)} tail levels)")
    return final_ranked


# Concurrency controls (initialized later)
query_semaphore = None
checkpoint_lock = None

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"\n[*] Model      : {args.model}")
print(f"[*] Batch size : {args.batch_size}  |  Batch top-K: {args.batch_top_k}  |  Max final merge: {args.max_final_merge}")
print(f"[*] Concurrency: {args.concurrency} queries in parallel")
print(f"[*] WITH SOLUTIONS in prompts (recursive merge-sort)")

print("\n[1/5] Loading corpus...")
corpus_data  = load_jsonl(dataset_dir / "corpus.jsonl")
doc_ids      = [d["_id"]  for d in corpus_data]
corpus_by_id = {d["_id"]: d for d in corpus_data}
print(f"  {len(doc_ids):,} documents")

print("\n[2/5] Loading solutions...")
solutions_by_id = load_solutions(args.solutions_file)
print(f"  {len(solutions_by_id):,} solutions loaded")
# Check coverage
covered = sum(1 for did in doc_ids if did in solutions_by_id)
print(f"  Coverage: {covered}/{len(doc_ids)} ({100*covered/len(doc_ids):.1f}%)")

print("\n[3/5] Loading queries & qrels...")
qrels_file   = dataset_dir / "qrels_pool.tsv"
queries_file = dataset_dir / "queries.jsonl"

qrels        = load_qrels(qrels_file)
queries_data = load_jsonl(queries_file)

query_ids   = [q["_id"] for q in queries_data if q["_id"] in qrels]
query_texts = {}
for q in queries_data:
    raw        = q["text"]
    marker     = "Given the following mathematical problem:\n\n"
    end_marker = "\n\nFind other mathematical problems"
    if marker in raw and end_marker in raw:
        start = raw.index(marker) + len(marker)
        end   = raw.index(end_marker)
        query_texts[q["_id"]] = raw[start:end].strip()
    else:
        query_texts[q["_id"]] = raw

# Build query solutions lookup (queries reference corpus problems)
query_solutions = {}
for q in queries_data:
    qid = q["_id"]
    # Query _id format is like "q00031" but we need the actual problem_id from metadata
    meta = q.get("metadata", {})
    problem_id = meta.get("problem_id", "")
    if problem_id:
        # Try to find solution
        for key, sol in solutions_by_id.items():
            if problem_id in key:
                query_solutions[qid] = sol
                break
    if qid not in query_solutions:
        query_solutions[qid] = "[Query solution not found]"

if args.limit:
    query_ids = query_ids[:args.limit]
    print(f"  Limiting to {args.limit} queries")
print(f"  {len(query_ids)} queries with qrels")

n_rel   = [len(v) for v in qrels.values()]
avg_rel = float(np.mean(n_rel))
print(f"  Avg relevant per query: {avg_rel:.1f}")

print("\n[4/5] Loading dense retrieval results...")
gemini_results  = load_dense_results(args.gemini)
qwen4b_results  = load_dense_results(args.qwen4b)
qwen06b_results = load_dense_results(args.qwen06b)
print(f"  Gemini: {len(gemini_results)} | Qwen4B: {len(qwen4b_results)} | Qwen0.6B: {len(qwen06b_results)}")

dense_results_list = [gemini_results, qwen4b_results, qwen06b_results]
top_k_list         = [args.gemini_top, args.qwen4b_top, args.qwen06b_top]

# ── Load exclusion list (query duplicates in corpus) ─────────────────────────

exclusion_file = dataset_dir / "per_query_excluded_ids.json"
excluded_ids = {}
if exclusion_file.exists():
    with open(exclusion_file) as f:
        excluded_ids = json.load(f)
    excluded_ids = {qid: set(ids) for qid, ids in excluded_ids.items()}
    print(f"  Loaded exclusions for {len(excluded_ids)} queries")

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
    print(f"\n  Loaded {len(cached)} cached results")

# ── Concurrent reranking ─────────────────────────────────────────────────────

async def process_single_query(qid, cache_file, pbar):
    """Process a single query with semaphore-limited concurrency."""
    global query_semaphore, checkpoint_lock

    async with query_semaphore:
        query_problem = query_texts[qid]
        query_solution = query_solutions.get(qid, "[Solution not available]")
        gold_ids = list(qrels.get(qid, {}).keys())
        query_excluded = excluded_ids.get(qid, set())

        candidates, candidate_scores, injected = get_union_candidates(
            qid, dense_results_list, top_k_list, gold_ids, query_excluded
        )

        ranked = await gpt_mergesortinter(
            query_problem, query_solution, candidates, corpus_by_id, solutions_by_id
        )

        rec = {
            "query_id":     qid,
            "ranked":       ranked,
            "candidates":   candidates,
            "injected":     injected,
            "dense_scores": candidate_scores,
        }

        # Thread-safe checkpoint writing
        async with checkpoint_lock:
            cache_file.write(json.dumps(rec) + "\n")
            cache_file.flush()

        pbar.update(1)
        await asyncio.sleep(args.sleep)

        return rec, bool(injected)


async def run_concurrent_reranking():
    """Run reranking for all queries concurrently."""
    global query_semaphore, checkpoint_lock

    query_semaphore = asyncio.Semaphore(args.concurrency)
    checkpoint_lock = asyncio.Lock()

    results_list = []
    n_injected = 0

    # Separate cached vs uncached queries
    uncached_qids = [qid for qid in query_ids if qid not in cached]
    cached_qids = [qid for qid in query_ids if qid in cached]

    # Add cached results first
    for qid in cached_qids:
        results_list.append(cached[qid])

    print(f"  {len(cached_qids)} cached, {len(uncached_qids)} to process")

    if uncached_qids:
        cache_file = open(ckpt_path, "a")

        with tqdm(total=len(uncached_qids), desc="GPT reranking") as pbar:
            tasks = [
                process_single_query(qid, cache_file, pbar)
                for qid in uncached_qids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cache_file.close()

        # Process results
        for result in results:
            if isinstance(result, Exception):
                print(f"  Warning: Query failed with {result}")
                continue
            rec, was_injected = result
            results_list.append(rec)
            if was_injected:
                n_injected += 1

    return results_list, n_injected


print(f"\n[5/5] Reranking with {args.model} (mergesortinter + solutions)...")
print(f"  Concurrency: {args.concurrency} queries in parallel")

results_list, n_injected = asyncio.run(run_concurrent_reranking())

# ── Build run dict ────────────────────────────────────────────────────────────

run = {}
for rec in results_list:
    qid          = rec["query_id"]
    ranked       = rec.get("ranked", [])
    candidates   = rec.get("candidates", [])
    dense_scores = rec.get("dense_scores", {})

    docs            = {}
    gpt_ranked_dids = set()
    total           = len(candidates)

    for item in ranked:
        did  = item["did"]
        rank = item["rank"]
        docs[did] = float(total - rank + 1)
        gpt_ranked_dids.add(did)

    tail_score = float(len(candidates) - len(gpt_ranked_dids))
    remaining  = sorted(
        [did for did in candidates if did not in gpt_ranked_dids],
        key=lambda d: -dense_scores.get(d, 0.0)
    )
    for did in remaining:
        docs[did] = tail_score
        tail_score -= 1.0

    run[qid] = docs

# ── Evaluate ──────────────────────────────────────────────────────────────────

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {"ndcg_cut.10,50,100", "recall.10,50,100", "recip_rank", "success.1,5,10"}
)
results = evaluator.evaluate(run)

def mean(key):
    return float(np.mean([v.get(key, 0.0) for v in results.values()]))

print(f"\n{'='*70}")
print(f"  {args.model}  —  mergesortinter + SOLUTIONS (recursive)")
print(f"  batch_size={args.batch_size}, batch_top_k={args.batch_top_k}, max_final_merge={args.max_final_merge}")
print(f"  Queries: {len(query_ids)}  |  Gold-injected: {n_injected}")
print(f"{'='*70}")
print(f"  {'Metric':<20} {'Score':>8}")
print(f"  {'-'*30}")
print(f"  {'MRR':<20} {mean('recip_rank'):>8.4f}")
print(f"  {'NDCG@10':<20} {mean('ndcg_cut_10'):>8.4f}")
print(f"  {'NDCG@50':<20} {mean('ndcg_cut_50'):>8.4f}")
print(f"  {'NDCG@100':<20} {mean('ndcg_cut_100'):>8.4f}")
print(f"  {'-'*30}")
print(f"  {'Recall@10':<20} {mean('recall_10'):>8.4f}")
print(f"  {'Recall@50':<20} {mean('recall_50'):>8.4f}")
print(f"  {'Recall@100':<20} {mean('recall_100'):>8.4f}")
print(f"  {'-'*30}")
print(f"  {'Success@1':<20} {mean('success_1'):>8.4f}")
print(f"  {'Success@5':<20} {mean('success_5'):>8.4f}")
print(f"  {'Success@10':<20} {mean('success_10'):>8.4f}")
print(f"{'='*70}")

# ── Save results ──────────────────────────────────────────────────────────────

out_path = dataset_dir / "results_gpt_mergesortinter_with_solutions.jsonl"
rows = []
for qid in query_ids:
    v = results.get(qid, {})
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked_list = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rows.append({
        "query_id":   qid,
        "n_relevant": len(rel_docs),
        "mrr":        round(v.get("recip_rank",   0), 4),
        "ndcg@10":    round(v.get("ndcg_cut_10",  0), 4),
        "ndcg@50":    round(v.get("ndcg_cut_50",  0), 4),
        "ndcg@100":   round(v.get("ndcg_cut_100", 0), 4),
        "recall@10":  round(v.get("recall_10",    0), 4),
        "recall@50":  round(v.get("recall_50",    0), 4),
        "recall@100": round(v.get("recall_100",   0), 4),
        "top10":      ranked_list,
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Results → {out_path}")
print(f"[+] Cache   → {ckpt_path}")
