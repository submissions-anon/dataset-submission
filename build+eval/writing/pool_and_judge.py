"""
pool_and_judge.py
=================
NDCG-Pooled construction for the Math Reasoning-Analogue benchmark.

Process:
  1. Load per-query result files from each retrieval model
  2. For each query, take top-K doc IDs from each model → union (deduplicated)
  3. Strip already-judged IDs (from qrels.tsv)
  4. GPT-5 judges unjudged candidates: does this problem share the same
     abstract reasoning pattern as the query?
  5. Append positive judgments → write qrels_pooled.tsv

Usage:
  export OPENAI_API_KEY=...
  python pool_and_judge.py \\
      --run-dir dataset \\
      --runs results_qwen3_Qwen_Qwen3_Embedding_0.6B_framed_full.jsonl \\
             results_qwen3_Qwen_Qwen3_Embedding_4B_framed_full.jsonl \\
             results_gemini_gemini_embedding_2_preview_framed_full.jsonl \\
      --pool-k 10
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--runs",        nargs="+", required=True,
                    help="Per-query result JSONL files from eval_math_analogues.py")
parser.add_argument("--run-dir",     default="dataset")
parser.add_argument("--pool-k",      type=int, default=10,
                    help="Top-K to take from each model per query (default: 10)")
parser.add_argument("--corpus",      default="dataset/corpus.jsonl")
parser.add_argument("--qrels",       default="dataset/qrels.tsv")
parser.add_argument("--queries",     default="dataset/queries.jsonl")
parser.add_argument("--out",         default="dataset/qrels_pooled.tsv")
parser.add_argument("--ckpt",        default="checkpoints/pool_judgments.json")
parser.add_argument("--judge-batch", type=int, default=15,
                    help="Candidates per GPT-5 call (math problems are longer than tweets)")
parser.add_argument("--model",       default="gpt-5")
parser.add_argument("--sleep",       type=float, default=0.3)
args = parser.parse_args()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Load data ─────────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

print("[1/5] Loading corpus...")
corpus = {}
for doc in load_jsonl(args.corpus):
    corpus[doc["_id"]] = doc
print(f"  {len(corpus):,} documents")

print("[2/5] Loading queries...")
queries = {}
for q in load_jsonl(args.queries):
    # Strip framing wrapper — give GPT just the raw problem statement
    raw        = q["text"]
    marker     = "Given the following mathematical problem:\n\n"
    end_marker = "\n\nFind other mathematical problems"
    if marker in raw and end_marker in raw:
        start = raw.index(marker) + len(marker)
        end   = raw.index(end_marker)
        queries[q["_id"]] = {
            "problem":    raw[start:end].strip(),
            "meta":       q.get("metadata", {}).get("meta_program", ""),
            "problem_id": q.get("metadata", {}).get("problem_id", q["_id"]),
        }
    else:
        queries[q["_id"]] = {
            "problem":    raw,
            "meta":       q.get("metadata", {}).get("meta_program", ""),
            "problem_id": q.get("metadata", {}).get("problem_id", q["_id"]),
        }
print(f"  {len(queries):,} queries")

print("[3/5] Loading existing qrels...")
existing_qrels = defaultdict(dict)
with open(args.qrels) as f:
    next(f)
    for line in f:
        line = line.strip()
        if not line:
            continue
        qid, did, score = line.split("\t")
        existing_qrels[qid][did] = int(score)
total_existing = sum(len(v) for v in existing_qrels.values())
print(f"  {total_existing:,} existing qrel pairs across {len(existing_qrels)} queries")

# ── Pool top-K from each model ────────────────────────────────────────────────

print("[4/5] Pooling candidates...")

run_dir = Path(args.run_dir)
pooled_candidates = defaultdict(set)

for run_file in args.runs:
    path = Path(run_file) if Path(run_file).is_absolute() else run_dir / run_file
    records = load_jsonl(path)
    n_with_ranked = 0
    for rec in records:
        qid    = rec["query_id"]
        ranked = rec.get("ranked")
        if not ranked:
            continue
        n_with_ranked += 1
        for did, score in ranked[:args.pool_k]:
            pooled_candidates[qid].add(str(did))
    print(f"  {path.name}: {n_with_ranked} queries with ranked lists")

if not any(pooled_candidates.values()):
    raise SystemExit("[!] No ranked lists found. Make sure result files contain 'ranked' field.")

# Strip already-judged and self
unjudged_per_query = {}
total_pool = total_unjudged = 0
for qid, candidates in pooled_candidates.items():
    q_info   = queries.get(qid, {})
    self_pid = q_info.get("problem_id", "")
    already  = set(existing_qrels.get(qid, {}).keys()) | {qid, self_pid}
    unjudged = candidates - already
    unjudged_per_query[qid] = sorted(unjudged)
    total_pool     += len(candidates)
    total_unjudged += len(unjudged)

print(f"  Total pooled candidates : {total_pool:,}")
print(f"  Already judged          : {total_pool - total_unjudged:,}")
print(f"  Unjudged to judge       : {total_unjudged:,}")

# ── GPT-5 judgment ────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are judging relevance for a mathematical reasoning-analogue retrieval benchmark.

You will receive:
- A QUERY PROBLEM and its reasoning fingerprint
- GOLD PROBLEMS confirmed to share its meta-program (for calibration)
- CANDIDATE PROBLEMS that are unjudged

For each candidate decide if it shares the same CORE REASONING PATTERN as the
query — the same abstract "aha moment" — regardless of topic, surface vocabulary,
or mathematical field.

Relevance grades:
  2 = SAME meta-program: a mathematician would immediately recognize the same
      eureka insight across completely different fields. Abstract proof move
      is structurally identical.
  0 = DIFFERENT reasoning pattern entirely.

Respond ONLY with a JSON array (one entry per candidate, in input order):
[{"id": "<problem_id>", "score": <0|2>}, ...]"""


def judge_batch(qid, q_info, gold_dids, candidate_dids):
    # Query block
    fp = corpus.get(qid, {}).get("metadata", {})
    query_block = (
        f"QUERY PROBLEM ({q_info['problem_id']}):\n"
        f"{q_info['problem'][:600]}\n\n"
        f"  Fingerprint  : {fp.get('fingerprint_summary', '')}\n"
        f"  Meta-strategy: {fp.get('meta_strategy', '')[:200]}"
    )

    # Gold context (up to 3 confirmed same-meta-program problems)
    gold_lines = []
    for did in gold_dids[:3]:
        doc = corpus.get(did, {})
        gfp = doc.get("metadata", {})
        gold_lines.append(
            f"  ID {did}\n"
            f"  Problem: {doc.get('text','')[:200]}\n"
            f"  Fingerprint: {gfp.get('fingerprint_summary', '')}"
        )
    gold_block = "\n\n".join(gold_lines) if gold_lines else "  (none — judge against query fingerprint only)"

    # Candidate block
    cand_lines = []
    for did in candidate_dids:
        doc = corpus.get(did, {})
        cfp = doc.get("metadata", {})
        cand_lines.append(
            f"  ID {did}\n"
            f"  Problem: {doc.get('text','')[:300]}\n"
            f"  Fingerprint: {cfp.get('fingerprint_summary', '')}"
        )

    if not cand_lines:
        return []

    user_block = (
        f"{query_block}\n\n"
        f"GOLD PROBLEMS (confirmed same meta-program):\n{gold_block}\n\n"
        f"CANDIDATE PROBLEMS TO JUDGE:\n" + "\n\n".join(cand_lines)
    )

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_block},
                ],
                temperature=1.0,
                max_completion_tokens=50000,
                response_format={"type": "json_object"},
            )
            raw    = resp.choices[0].message.content
            parsed = json.loads(raw)
            items  = parsed if isinstance(parsed, list) else next(
                (v for v in parsed.values() if isinstance(v, list)), []
            )
            return [r for r in items if isinstance(r, dict)]
        except Exception as e:
            print(f"  [judge] error attempt {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return []


print("[5/5] Judging unjudged candidates...")

ckpt_path = Path(args.ckpt)
ckpt_path.parent.mkdir(exist_ok=True)
new_judgments = {}
if ckpt_path.exists():
    new_judgments = json.loads(ckpt_path.read_text())
    done = sum(1 for v in new_judgments.values() if v != "FAILED")
    print(f"  Resuming: {done} queries already judged")

to_judge = [
    qid for qid in unjudged_per_query
    if qid not in new_judgments or new_judgments.get(qid) == "FAILED"
]
print(f"  {len(to_judge)} queries to judge")

for qid in tqdm(to_judge, desc="  Judging"):
    candidates = unjudged_per_query[qid]
    if not candidates:
        new_judgments[qid] = []
        continue

    q_info    = queries.get(qid, {"problem": "", "meta": "", "problem_id": qid})
    gold_dids = [did for did, s in existing_qrels.get(qid, {}).items() if s >= 2]

    all_results = []
    batch_ok    = True
    for i in range(0, len(candidates), args.judge_batch):
        batch   = candidates[i:i+args.judge_batch]
        results = judge_batch(qid, q_info, gold_dids, batch)
        if results:
            all_results.extend(results)
        else:
            batch_ok = False
        time.sleep(args.sleep)

    new_judgments[qid] = "FAILED" if (not all_results and not batch_ok) else all_results
    ckpt_path.write_text(json.dumps(new_judgments))
    time.sleep(args.sleep)

# ── Write pooled qrels ────────────────────────────────────────────────────────

out_path = Path(args.out)
added = skipped_failed = 0

with open(out_path, "w") as f:
    f.write("query-id\tcorpus-id\tscore\n")

    # Existing qrels first
    for qid, docs in existing_qrels.items():
        for did, score in docs.items():
            f.write(f"{qid}\t{did}\t{score}\n")

    # New positive judgments
    for qid, judgments in new_judgments.items():
        if not isinstance(judgments, list):
            skipped_failed += 1
            continue
        existing = existing_qrels.get(qid, {})
        for r in judgments:
            did   = str(r.get("id", ""))
            score = int(r.get("score", 0))
            if score > 0 and did and did not in existing:
                f.write(f"{qid}\t{did}\t{score}\n")
                added += 1

print(f"\nDone.")
print(f"  Original qrels : {total_existing:,}")
print(f"  New positives  : {added:,}")
print(f"  Failed queries : {skipped_failed}")
print(f"  Total pooled   : {total_existing + added:,}")
print(f"  Output         : {out_path}")
