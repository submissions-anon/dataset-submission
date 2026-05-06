"""
eval_gemini_v2.py
=================
Gemini Embedding dense retrieval for v2 Writing Analogues benchmark.

Changes from v1:
  - Single binary qrels file (qrels.tsv). No Easy/Hard/Graded split.
  - No subfield taxonomy. No subfield_map, no per-(author, subfield) breakdown.
  - --benchmark_dir points to build_benchmark_v2.py output
    (corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json live here).
  - Merged corpus_track accepts v2 gold CSV OR (optionally) the distractor CSV
    for richer per-author breakdown on pool authors too.

Requires:
  pip install google-genai pytrec_eval tqdm numpy
  export GEMINI_API_KEY=...

Usage:
  python eval_gemini_v2.py \
      --benchmark_dir benchmark_v2/ \
      --corpus_dir corpus/ \
      --gold_track corpus_track.csv \
      --tag v2
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("pip install google-genai")

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark_dir", required=True,
                    help="Directory containing corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json")
parser.add_argument("--corpus_dir",    required=True,
                    help="Directory containing N.txt files")
parser.add_argument("--gold_track",    default="corpus_track.csv",
                    help="Path to v2 gold CSV (snippet_id, author_name, post_title, post_url)")
parser.add_argument("--tag",           default="v2",
                    help="Suffix appended to output filenames.")
parser.add_argument("--model",         default="gemini-embedding-2-preview")
parser.add_argument("--dim",           type=int, default=3072,
                    help="Output dimensionality (3072 / 1536 / 768)")
parser.add_argument("--batch_size",    type=int, default=50)
parser.add_argument("--sleep",         type=float, default=1.0)
parser.add_argument("--top_k",         type=int, default=1000)
parser.add_argument("--chunk_size",    type=int, default=500)
parser.add_argument("--ckpt_dir",      default=None)
args = parser.parse_args()

bench_dir = Path(args.benchmark_dir)
ckpt_dir  = Path(args.ckpt_dir) if args.ckpt_dir else bench_dir / "gemini_cache"
ckpt_dir.mkdir(exist_ok=True, parents=True)
model_slug = args.model.replace("/", "_").replace("-", "_")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Set GEMINI_API_KEY environment variable")
client = genai.Client(api_key=api_key)


# ── Load benchmark ────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d['_id']] = d
    return out

print("[1/5] Loading benchmark...")
corpus  = load_jsonl(bench_dir / "corpus.jsonl")
queries = load_jsonl(bench_dir / "queries.jsonl")

corpus_dir = Path(args.corpus_dir)
for sid, doc in corpus.items():
    doc['text'] = (corpus_dir / f"{sid}.txt").read_text(encoding="utf-8", errors="replace").strip()
for sid, q in queries.items():
    q['text'] = (corpus_dir / f"{sid}.txt").read_text(encoding="utf-8", errors="replace").strip()

# Binary qrels
qrels = {}
with open(bench_dir / "qrels.tsv") as f:
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
print(f"  Qrels: {sum(len(v) for v in qrels.values())} pairs across {len(qrels)} queries")

with open(bench_dir / "per_query_excluded_ids.json") as f:
    excluded = json.load(f)
excluded = {qid: set(str(x) for x in ids) for qid, ids in excluded.items()}

# Per-author metadata (for breakdown)
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

corpus_ids   = list(corpus.keys())
corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
cid_to_idx   = {cid: i for i, cid in enumerate(corpus_ids)}
query_ids    = list(queries.keys())
query_texts  = [queries[qid]['text'] for qid in query_ids]

print(f"  Corpus:  {len(corpus_ids)} snippets")
print(f"  Queries: {len(query_ids)}")

# ── Embed with checkpointing ──────────────────────────────────────────────────

def embed_batch(texts, task_type):
    retries = 5
    for attempt in range(retries):
        try:
            resp = client.models.embed_content(
                model=args.model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=args.dim,
                ),
            )
            return np.array([e.values for e in resp.embeddings], dtype=np.float32)
        except Exception as e:
            wait = 2 * (2 ** attempt)
            print(f"\n  [attempt {attempt+1}/{retries}, retry in {wait}s] {e}")
            time.sleep(wait)
    raise RuntimeError("Gemini embedding failed after 5 retries")


def embed_with_checkpoint(texts, ckpt_prefix, task_type):
    total  = len(texts)
    chunks = list(range(0, total, args.chunk_size))
    out    = []
    last_done = -1

    for chunk_start in chunks:
        p = ckpt_dir / f"{ckpt_prefix}_{model_slug}_chunk{chunk_start}.npy"
        if p.exists():
            out.append(np.load(p))
            last_done = chunk_start
        else:
            break

    if out and last_done == chunks[-1]:
        print(f"  Loaded all {len(chunks)} chunks from cache")
        return np.concatenate(out, axis=0)

    resume_from = chunks[len(out)] if out else 0
    if out:
        print(f"  Resuming from chunk {resume_from} ({len(out)} cached)")

    for chunk_start in tqdm(chunks, desc=f"Embedding {ckpt_prefix}"):
        if chunk_start < resume_from:
            continue
        chunk_texts = texts[chunk_start: chunk_start + args.chunk_size]
        chunk_embs = []
        for i in range(0, len(chunk_texts), args.batch_size):
            batch = chunk_texts[i: i + args.batch_size]
            chunk_embs.append(embed_batch(batch, task_type))
            time.sleep(args.sleep)
        chunk_arr = np.concatenate(chunk_embs, axis=0)
        # L2 normalize
        norms = np.linalg.norm(chunk_arr, axis=1, keepdims=True)
        chunk_arr = chunk_arr / np.clip(norms, 1e-12, None)
        np.save(ckpt_dir / f"{ckpt_prefix}_{model_slug}_chunk{chunk_start}.npy", chunk_arr)
        out.append(chunk_arr)

    return np.concatenate(out, axis=0)

print(f"\n[2/5] Embedding corpus...")
corp_embs = embed_with_checkpoint(corpus_texts, "corpus", "RETRIEVAL_DOCUMENT")

# Sanity check: cached embeddings must align with current corpus_ids
if corp_embs.shape[0] != len(corpus_ids):
    raise SystemExit(
        f"\n[error] Corpus embedding cache has {corp_embs.shape[0]} rows but "
        f"current corpus has {len(corpus_ids)} snippets.\n"
        f"The cache is stale (likely because the corpus was modified, e.g. via "
        f"demote_authors.py).\n"
        f"Fix: rm {ckpt_dir}/corpus_*.npy  and rerun."
    )

print(f"\n[3/5] Embedding queries...")
# Embed raw query text — no instruction prefix
query_embs = embed_with_checkpoint(query_texts, "queries", "RETRIEVAL_QUERY")

# Same sanity check for queries
if query_embs.shape[0] != len(query_ids):
    raise SystemExit(
        f"\n[error] Query embedding cache has {query_embs.shape[0]} rows but "
        f"current benchmark has {len(query_ids)} queries.\n"
        f"Fix: rm {ckpt_dir}/queries_*.npy  and rerun."
    )

# ── Retrieve with per-query exclusion ─────────────────────────────────────────

print(f"\n[4/5] Retrieving (top_k={args.top_k})...")
SCORE_BATCH = 256
run = {}

for i in tqdm(range(0, len(query_ids), SCORE_BATCH), desc="Scoring"):
    q_batch = query_embs[i: i + SCORE_BATCH]
    scores  = q_batch @ corp_embs.T
    for j, qid in enumerate(query_ids[i: i + SCORE_BATCH]):
        row  = scores[j].copy()
        excl = excluded.get(qid, set())
        for cid in excl:
            idx = cid_to_idx.get(cid)
            if idx is not None:
                row[idx] = -np.inf
        top_idx  = np.argsort(row)[::-1][: args.top_k]
        run[qid] = {corpus_ids[k]: float(row[k]) for k in top_idx if row[k] > -np.inf}

# ── Evaluate ──────────────────────────────────────────────────────────────────

k_values = [10, 50, 100, 1000]
metrics  = set()
for k in k_values:
    metrics.add(f"ndcg_cut_{k}")
    metrics.add(f"recall_{k}")
metrics.add("map")

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
per_query = evaluator.evaluate(run)

def mean_metric(key):
    return float(np.mean([v.get(key, 0.0) for v in per_query.values()])) if per_query else 0.0

ordered = (
    [f"ndcg_cut_{k}" for k in k_values] +
    [f"recall_{k}"   for k in k_values] +
    ["map"]
)

print(f"\n{'='*60}")
print(f"  Gemini Embedding — Writing Analogues v2")
print(f"{'='*60}")
print(f"  Model:   {args.model}")
print(f"  Corpus:  {len(corpus_ids)} snippets")
print(f"  Queries: {len(query_ids)}")
print(f"  {'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8}")
print(f"  {'-'*46}")
for metric in ordered:
    vals = [v.get(metric, 0.0) for v in per_query.values()]
    print(f"  {metric:<20} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

# Per-author breakdown
if snippet_author:
    author_ndcg10 = defaultdict(list)
    author_ndcg50 = defaultdict(list)
    for qid, md in per_query.items():
        a = snippet_author.get(qid, "unknown")
        author_ndcg10[a].append(md.get("ndcg_cut_10", 0.0))
        author_ndcg50[a].append(md.get("ndcg_cut_50", 0.0))

    print(f"\n  {'─'*56}")
    print(f"  Per-author breakdown")
    print(f"  {'─'*56}")
    print(f"  {'Author':<30} {'nDCG@10':>8} {'nDCG@50':>8} {'n':>5}")
    print(f"  {'─'*56}")
    for author in sorted(author_ndcg10, key=lambda a: -np.mean(author_ndcg10[a])):
        v10 = author_ndcg10[author]
        v50 = author_ndcg50[author]
        print(f"  {author[:29]:<30} {np.mean(v10):>8.4f} {np.mean(v50):>8.4f} {len(v10):>5}")

# ── Save outputs ──────────────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""
results_path = bench_dir / f"gemini_{model_slug}{tag_suffix}_results.jsonl"
rows_out = []
for qid in query_ids:
    v = per_query.get(qid, {})
    qrun = run.get(qid, {})
    ranked_list = sorted(qrun.keys(), key=lambda cid: qrun[cid], reverse=True)
    rows_out.append({
        "query_id":    qid,
        "author":      snippet_author.get(qid, "unknown"),
        "ndcg@10":     round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50":     round(v.get("ndcg_cut_50", 0), 4),
        "recall@10":   round(v.get("recall_10",   0), 4),
        "recall@50":   round(v.get("recall_50",   0), 4),
        "recall@100":  round(v.get("recall_100",  0), 4),
        "map":         round(v.get("map",          0), 4),
        "ranked_list": ranked_list,
    })
results_path.write_text("\n".join(json.dumps(r) for r in rows_out))
print(f"\n[5/5] Results written:")
print(f"  {results_path}")

summary_path = bench_dir / f"gemini_{model_slug}{tag_suffix}_summary.json"
summary_path.write_text(json.dumps({
    "model":       args.model,
    "tag":         args.tag,
    "corpus_size": len(corpus_ids),
    "n_queries":   len(query_ids),
    "metrics":     {m: round(mean_metric(m), 4) for m in ordered},
    "per_author_ndcg10": {a: round(float(np.mean(v)), 4) for a, v in author_ndcg10.items()} if snippet_author else {},
}, indent=2))
print(f"  {summary_path}")
