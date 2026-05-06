"""
eval_lateon_v2.py
=================
ColBERT (multi-vector) retrieval eval for v2 Writing Analogues benchmark
using LateOn via PyLate + FastPLAID.

Corpus texts live as individual .txt files in --corpus_dir.
Benchmark metadata (corpus.jsonl, queries.jsonl, qrels.tsv,
per_query_excluded_ids.json) lives in --benchmark_dir.

Per-author breakdown from --gold_track CSV.

Requirements:
  pip install -U pylate pytrec_eval tqdm numpy

Usage:
  python eval_lateon_v2.py \
      --benchmark_dir benchmark_v2/ \
      --corpus_dir corpus/ \
      --gold_track corpus_track.csv \
      --tag v2
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm

from pylate import indexes, models, retrieve

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark_dir", required=True,
                    help="Directory containing corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json")
parser.add_argument("--corpus_dir",    required=True,
                    help="Directory containing N.txt files")
parser.add_argument("--gold_track",    default="corpus_track.csv",
                    help="Path to v2 gold CSV (snippet_id, author_name, post_title, post_url)")
parser.add_argument("--tag",           default="v2",
                    help="Suffix appended to output filenames")
parser.add_argument("--model",         default="lightonai/LateOn")
parser.add_argument("--batch_size",    type=int, default=32)
parser.add_argument("--top_k",         type=int, default=1000)
parser.add_argument("--index_dir",     default="lateon_plaid_index_writing",
                    help="Directory for the PLAID index")
parser.add_argument("--rebuild_index", action="store_true",
                    help="Force rebuild the PLAID index even if it exists")
parser.add_argument("--device",        default=None,
                    help="Device (auto-detected if omitted)")
args = parser.parse_args()

bench_dir  = Path(args.benchmark_dir)
corpus_dir = Path(args.corpus_dir)
model_slug = args.model.replace("/", "_").replace("-", "_")
index_name = f"{model_slug}_writing_v2"

# ── Load benchmark ────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d
    return out

print("[1/5] Loading benchmark...")
corpus  = load_jsonl(bench_dir / "corpus.jsonl")
queries = load_jsonl(bench_dir / "queries.jsonl")

# Load actual text from .txt files
for sid, doc in corpus.items():
    doc["text"] = (corpus_dir / f"{sid}.txt").read_text(encoding="utf-8", errors="replace").strip()
for sid, q in queries.items():
    q["text"] = (corpus_dir / f"{sid}.txt").read_text(encoding="utf-8", errors="replace").strip()

# Binary qrels
qrels = {}
with open(bench_dir / "qrels.tsv") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            qid, cid, score = parts
        elif len(parts) == 4:
            qid, _, cid, score = parts
        else:
            continue
        qrels.setdefault(qid, {})[cid] = int(score)
print(f"  Qrels: {sum(len(v) for v in qrels.values())} pairs across {len(qrels)} queries")

# Per-query exclusions
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
corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
query_ids    = list(queries.keys())
query_texts  = [queries[qid]["text"] for qid in query_ids]

print(f"  Corpus:  {len(corpus_ids)} snippets")
print(f"  Queries: {len(query_ids)}")

# ── Sanity check: show what's actually being fed ──────────────────────────────

corpus_char_lens = [len(t) for t in corpus_texts]
query_char_lens  = [len(t) for t in query_texts]
empty_corpus = [cid for cid, t in zip(corpus_ids, corpus_texts) if not t.strip()]
empty_queries = [qid for qid, t in zip(query_ids, query_texts) if not t.strip()]

print(f"\n  ── Text sanity check ──")
print(f"  Corpus char lengths:  min={min(corpus_char_lens)}  median={int(np.median(corpus_char_lens))}  "
      f"max={max(corpus_char_lens)}  mean={int(np.mean(corpus_char_lens))}")
print(f"  Query char lengths:   min={min(query_char_lens)}  median={int(np.median(query_char_lens))}  "
      f"max={max(query_char_lens)}  mean={int(np.mean(query_char_lens))}")
if empty_corpus:
    print(f"  ⚠ {len(empty_corpus)} EMPTY corpus texts: {empty_corpus[:5]}")
if empty_queries:
    print(f"  ⚠ {len(empty_queries)} EMPTY query texts: {empty_queries[:5]}")

print(f"\n  Sample corpus docs:")
for i in range(min(3, len(corpus_ids))):
    cid = corpus_ids[i]
    txt = corpus_texts[i]
    author = snippet_author.get(cid, "?")
    print(f"    [{cid}] (author={author}, {len(txt)} chars) {txt[:120]}...")

print(f"\n  Sample queries:")
for i in range(min(3, len(query_ids))):
    qid = query_ids[i]
    txt = query_texts[i]
    author = snippet_author.get(qid, "?")
    n_rel = len(qrels.get(qid, {}))
    print(f"    [{qid}] (author={author}, {len(txt)} chars, {n_rel} rels) {txt[:120]}...")

# ── Load model ────────────────────────────────────────────────────────────────

print(f"\n[2/5] Loading ColBERT model ({args.model})...")
model_kwargs = {"model_name_or_path": args.model}
if args.device:
    model_kwargs["device"] = args.device
model_kwargs["document_length"] = 8192
model = models.ColBERT(**model_kwargs)

# ── Build or load PLAID index ─────────────────────────────────────────────────

index_folder = Path(args.index_dir)
index_exists = (index_folder / index_name).exists() and not args.rebuild_index

if index_exists:
    print(f"\n[3/5] Loading existing PLAID index from {index_folder / index_name}...")
    index = indexes.PLAID(
        index_folder=str(index_folder),
        index_name=index_name,
    )
else:
    print(f"\n[3/5] Building PLAID index ({len(corpus_ids)} snippets)...")
    index = indexes.PLAID(
        index_folder=str(index_folder),
        index_name=index_name,
        override=True,
    )

    doc_embeddings = model.encode(
        corpus_texts,
        batch_size=args.batch_size,
        is_query=False,
        show_progress_bar=True,
    )

    index.add_documents(
        documents_ids=corpus_ids,
        documents_embeddings=doc_embeddings,
    )
    del doc_embeddings
    print(f"  Index saved to {index_folder / index_name}")

# ── Retrieve ──────────────────────────────────────────────────────────────────

print(f"\n[4/5] Encoding queries & retrieving (top_k={args.top_k})...")

query_embeddings = model.encode(
    query_texts,
    batch_size=args.batch_size,
    is_query=True,
    show_progress_bar=True,
)

# Retrieve extra to compensate for exclusion filtering
max_excluded = max((len(excluded.get(qid, [])) for qid in query_ids), default=0)
retrieve_k   = min(args.top_k + max_excluded + 10, len(corpus_ids))

retriever = retrieve.ColBERT(index=index)

# Retrieve per-query with progress bar (MaxSim is the bottleneck)
run = {}
for qi in tqdm(range(len(query_ids)), desc="Retrieving"):
    qid = query_ids[qi]
    r = retriever.retrieve(
        queries_embeddings=[query_embeddings[qi]],
        k=retrieve_k,
    )
    query_excluded = excluded.get(qid, set())
    docs = {}
    for hit in r[0]:
        did = hit["id"]
        if did in query_excluded:
            continue
        docs[did] = hit["score"]
        if len(docs) >= args.top_k:
            break
    run[qid] = docs

# ── Evaluate ──────────────────────────────────────────────────────────────────

print(f"\n[5/5] Evaluating...")

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
print(f"  LateOn ColBERT — Writing Analogues v2")
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
author_ndcg10 = defaultdict(list)
author_ndcg50 = defaultdict(list)
for qid, md in per_query.items():
    a = snippet_author.get(qid, "unknown")
    author_ndcg10[a].append(md.get("ndcg_cut_10", 0.0))
    author_ndcg50[a].append(md.get("ndcg_cut_50", 0.0))

if snippet_author:
    print(f"\n  {'─'*56}")
    print(f"  Per-author breakdown")
    print(f"  {'─'*56}")
    print(f"  {'Author':<30} {'nDCG@10':>8} {'nDCG@50':>8} {'n':>5}")
    print(f"  {'─'*56}")
    for author in sorted(author_ndcg10, key=lambda a: -np.mean(author_ndcg10[a])):
        v10 = author_ndcg10[author]
        v50 = author_ndcg50[author]
        print(f"  {author[:29]:<30} {np.mean(v10):>8.4f} {np.mean(v50):>8.4f} {len(v10):>5}")

print(f"{'='*60}")

# ── Save outputs ──────────────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""
results_path = bench_dir / f"lateon_{model_slug}{tag_suffix}_results.jsonl"
rows_out = []
for qid in query_ids:
    v    = per_query.get(qid, {})
    qrun = run.get(qid, {})
    ranked_pairs = sorted(qrun.items(), key=lambda x: -x[1])
    rows_out.append({
        "query_id":    qid,
        "author":      snippet_author.get(qid, "unknown"),
        "ndcg@10":     round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50":     round(v.get("ndcg_cut_50", 0), 4),
        "recall@10":   round(v.get("recall_10",   0), 4),
        "recall@50":   round(v.get("recall_50",   0), 4),
        "recall@100":  round(v.get("recall_100",  0), 4),
        "map":         round(v.get("map",          0), 4),
        "ranked_list": [cid for cid, _ in ranked_pairs],
        "ranked":      ranked_pairs[:args.top_k],
    })

results_path.write_text("\n".join(json.dumps(r) for r in rows_out))
print(f"\n[+] Per-query results → {results_path}")

summary_path = bench_dir / f"lateon_{model_slug}{tag_suffix}_summary.json"
summary_path.write_text(json.dumps({
    "model":       args.model,
    "tag":         args.tag,
    "corpus_size": len(corpus_ids),
    "n_queries":   len(query_ids),
    "metrics":     {m: round(mean_metric(m), 4) for m in ordered},
    "per_author_ndcg10": {
        a: round(float(np.mean(v)), 4)
        for a, v in author_ndcg10.items()
    } if snippet_author else {},
}, indent=2))
print(f"[+] Summary           → {summary_path}")
