"""
eval_gpt_mergesortinter_think_first_recursive.py
=================================================
GPT reranking with RECURSIVE merge-sort-interleave for Math Reasoning-Analogue benchmark.

KEY DIFFERENCE: Two-stage "think-first" approach where the model:
  1. First reasons through how it would solve the query problem
  2. Then reasons through each candidate's likely solution approach
  3. Finally ranks based on shared abstract proof strategies

RECURSIVE VERSION: Handles large candidate pools by recursively applying
merge-sort-interleave until the merge pool is small enough for direct ranking.

Algorithm (recursive mergesortinter):
  1. Load top-N from Gemini, top-M from each Qwen + gold injection
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

Usage:
  export OPENAI_API_KEY=...
  python eval_gpt_mergesortinter_think_first_recursive.py
  python eval_gpt_mergesortinter_think_first_recursive.py --limit 5
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


def fix_json_backslashes(text):
    """
    Fix invalid JSON escape sequences from LaTeX notation.
    Escape all backslashes so LaTeX commands parse as literal text.
    """
    return text.replace('\\', '\\\\')


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir",  default="dataset")
parser.add_argument("--model",        default="gpt-5.2")
parser.add_argument("--gemini",       default="")
parser.add_argument("--qwen4b",       default="")
parser.add_argument("--qwen06b",      default="")
parser.add_argument("--gemini-top",   type=int, default=165)
parser.add_argument("--qwen4b-top",   type=int, default=85)
parser.add_argument("--qwen06b-top",  type=int, default=85)
parser.add_argument("--batch-size",   type=int, default=6,
                    help="Candidates per batch")
parser.add_argument("--batch-top-k",  type=int, default=2,
                    help="Top-k from each batch for merge pool")
parser.add_argument("--max-final-merge", type=int, default=16,
                    help="Max candidates for direct ranking (triggers recursion if exceeded)")
parser.add_argument("--ckpt",         default="checkpoints/gpt_mergesortinter_think_first_recursive.jsonl")
parser.add_argument("--sleep",        type=float, default=0.3)
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--limit",        type=int, default=None)
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

IMPORTANT: You must work through this in TWO STAGES:

STAGE 1 - THINK FIRST:
Before ranking, you MUST first reason through how each problem would be solved.
For the query and each candidate:
  - What's the key insight or "aha moment"?
  - What proof technique would you use?
  - What's the abstract meta-move?

STAGE 2 - RANK:
Only after thinking through each problem's solution approach should you rank
candidates by whether they share the query's abstract proof strategy.

WHAT "SAME ABSTRACT STRATEGY" MEANS:

Two problems match if their solutions would use the same high-level meta-move:
  - "Transform representation → swap order of operations → collapse"
  - "Exploit uniqueness of X to constrain the answer"
  - "Diagonalize/decompose → solve per component → reassemble"
  - "Sign-reversing involution for cancellation"
  - "Symmetry averaging to project onto invariants"
  - "Construct auxiliary object that converts to known form"

Problems can share a meta-move despite having NOTHING in common on the surface:
  - A metric space problem and a convex bodies problem might both use
    "uniqueness of a canonical object" as the key insight
  - An integral and a combinatorial sum might both use "swap order then collapse"

IGNORE surface similarity:
  ✗ Same mathematical field (analysis, algebra, geometry)
  ✗ Same objects (integrals, groups, matrices)
  ✗ Similar notation or formulas
  ✗ Problems that "look alike"

FOCUS ON the deep proof skeleton:
  ✓ What is the KEY INSIGHT that unlocks each problem?
  ✓ What meta-level trick or principle would the solution rely on?
  ✓ Would the same proof OUTLINE work for both problems?

Return JSON with your thinking AND ranking:
{{
  "query_analysis": {{
    "key_insight": "<1-2 sentences: what's the key insight for solving this?>",
    "meta_move": "<brief label for the abstract strategy>"
  }},
  "candidate_analyses": [
    {{
      "candidate_num": <int>,
      "key_insight": "<1-2 sentences: how would you solve this?>",
      "meta_move": "<brief label>"
    }},
    ...
  ],
  "ranked": [
    {{"candidate_num": <int>, "reason": "<why this shares the query's meta-move>"}},
    ...
  ]
}}"""

USER_TEMPLATE = """\
QUERY PROBLEM:
══════════════
{query_problem}

══════════════════════════════════════════════════════════════════════════════

CANDIDATE PROBLEMS:
{candidates}

══════════════════════════════════════════════════════════════════════════════

INSTRUCTIONS:
1. THINK FIRST about the query: How would YOU solve it? What's the key insight?
2. THINK about each candidate: How would each be solved? What's each key insight?
3. Only THEN rank by whether the ABSTRACT PROOF STRATEGY matches the query's.

Take your time reasoning through each problem before ranking. The goal is to
identify shared proof skeletons, not surface-level topic similarity."""

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

async def gpt_rerank_batch_async(query_problem_text, candidates, corpus_by_id):
    """Rerank a single batch of candidates with think-first prompting."""
    if not candidates:
        return []

    if len(candidates) == 1:
        return [{"did": candidates[0], "rank": 1, "reason": "single-candidate"}]

    num_to_did = {i + 1: did for i, did in enumerate(candidates)}

    def format_candidate(i, did):
        if did not in corpus_by_id:
            return None
        return f"[{i+1}] PROBLEM: {corpus_by_id[did]['text']}"

    candidate_lines = "\n\n".join(
        line for line in (format_candidate(i, did) for i, did in enumerate(candidates))
        if line is not None
    )

    system = SYSTEM_PROMPT.format(k=len(candidates))
    prompt = USER_TEMPLATE.format(
        query_problem=query_problem_text,
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
            # Fix LaTeX backslashes before JSON parsing
            clean = fix_json_backslashes(clean.strip())

            # Find JSON object
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

            # Also capture the thinking for potential analysis
            query_analysis = parsed.get("query_analysis", {})
            candidate_analyses = parsed.get("candidate_analyses", [])

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

            # Add any missing candidates at the end
            for did in candidates:
                if did not in {item["did"] for item in out}:
                    out.append({
                        "did":    did,
                        "rank":   len(out) + 1,
                        "reason": "missing-from-response",
                    })

            return out

        except KeyboardInterrupt:
            raise
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"\n  [batch attempt {attempt+1}/5, retry in {wait}s] {e}")
            await asyncio.sleep(wait)

    # Fallback: return in original order
    return [{"did": did, "rank": i+1, "reason": "fallback"} for i, did in enumerate(candidates)]


# ── Recursive MergeSortInter algorithm ───────────────────────────────────────

async def gpt_mergesortinter_recursive(query_problem_text, candidates, corpus_by_id,
                                        depth=0, all_tails=None):
    """
    Recursive merge-sort-interleave reranking with think-first prompting.

    Key idea: If the merge pool is still too large after one round of batching,
    recursively apply the same procedure until we get down to max_final_merge.

    Tails are collected at each level and interleaved at the end in reverse depth
    order (deepest tails first, then shallower tails).
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
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        result = await gpt_rerank_batch_async(query_problem_text, shuffled, corpus_by_id)
        return result, all_tails

    # Shuffle and batch
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    batches = [
        shuffled[i:i + batch_size]
        for i in range(0, len(shuffled), batch_size)
    ]
    n_batches = len(batches)
    print(f"    {indent}[depth={depth}] {len(candidates)} candidates -> {n_batches} batches of <={batch_size}")

    # Sort each batch in parallel
    tasks = [
        gpt_rerank_batch_async(query_problem_text, batch, corpus_by_id)
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
        query_problem_text, merge_pool, corpus_by_id,
        depth=depth + 1, all_tails=all_tails
    )


async def gpt_mergesortinter(query_problem_text, candidates, corpus_by_id):
    """
    Entry point for recursive merge-sort-interleave with think-first prompting.
    Returns list of ranked items with did, rank, reason.
    """
    print(f"    [mergesortinter-think-recursive] Starting with {len(candidates)} candidates")

    if not candidates:
        return []

    # Get recursive ranking result and all collected tails
    ranked_result, all_tails = await gpt_mergesortinter_recursive(
        query_problem_text, candidates, corpus_by_id
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

    print(f"    [mergesortinter-think-recursive] Final ranking: {len(final_ranked)} docs ({len(all_tails)} tail levels)")
    return final_ranked


def run_async(coro):
    """Run async coroutine in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"\n[*] Model      : {args.model}")
print(f"[*] Algorithm  : mergesortinter-RECURSIVE + THINK-FIRST")
print(f"[*] Params     : batch_size={args.batch_size}, batch_top_k={args.batch_top_k}, max_final_merge={args.max_final_merge}")

print("\n[1/4] Loading corpus...")
corpus_data  = load_jsonl(dataset_dir / "corpus.jsonl")
doc_ids      = [d["_id"]  for d in corpus_data]
corpus_by_id = {d["_id"]: d for d in corpus_data}
print(f"  {len(doc_ids):,} documents")

print("\n[2/4] Loading queries & qrels...")
qrels_file   = dataset_dir / "qrels_final.tsv"
queries_file = dataset_dir / "queries_final.jsonl"

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

if args.limit:
    query_ids = query_ids[:args.limit]
    print(f"  Limiting to {args.limit} queries")
print(f"  {len(query_ids)} queries with qrels")

n_rel   = [len(v) for v in qrels.values()]
avg_rel = float(np.mean(n_rel))
print(f"  Avg relevant per query: {avg_rel:.1f}")

print("\n[3/4] Loading dense retrieval results...")
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

# ── Reranking loop ────────────────────────────────────────────────────────────

print(f"\n[4/4] Reranking with {args.model} (mergesortinter-RECURSIVE + think-first)...")

results_list = []
cache_file   = open(ckpt_path, "a")
n_injected   = 0
total_candidates = []

for qid in tqdm(query_ids, desc="GPT reranking"):
    if qid in cached:
        results_list.append(cached[qid])
        continue

    query_problem = query_texts[qid]
    gold_ids      = list(qrels.get(qid, {}).keys())
    query_excluded = excluded_ids.get(qid, set())

    candidates, candidate_scores, injected = get_union_candidates(
        qid, dense_results_list, top_k_list, gold_ids, query_excluded
    )
    total_candidates.append(len(candidates))
    if injected:
        n_injected += 1

    print(f"\n  [{qid}] {len(candidates)} candidates")

    ranked = run_async(gpt_mergesortinter(query_problem, candidates, corpus_by_id))

    rec = {
        "query_id":     qid,
        "ranked":       ranked,
        "candidates":   candidates,
        "injected":     injected,
        "dense_scores": candidate_scores,
    }
    results_list.append(rec)
    cache_file.write(json.dumps(rec) + "\n")
    cache_file.flush()
    time.sleep(args.sleep)

cache_file.close()
if total_candidates:
    print(f"\n  Avg candidates per query: {np.mean(total_candidates):.1f}")

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
print(f"  {args.model}  —  mergesortinter-RECURSIVE + THINK-FIRST")
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

out_path = dataset_dir / "results_gpt_mergesortinter_think_first_recursive.jsonl"
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
        "top10":      ranked_list[:10],
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Results -> {out_path}")
print(f"[+] Cache   -> {ckpt_path}")
