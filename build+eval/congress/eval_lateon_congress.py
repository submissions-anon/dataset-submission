"""
eval_lateon_congress.py
========================
ColBERT (multi-vector) retrieval eval for the Congressional Hearing ToT benchmark
using LateOn via PyLate + FastPLAID.

Reads BEIR format directly — no separate txt files needed.

Requirements:
  pip install -U pylate pytrec_eval tqdm numpy

Usage:
  python eval_lateon_congress.py --benchmark_dir congress_corpus_data/beir_export/
  python eval_lateon_congress.py --benchmark_dir congress_corpus_data/beir_export/ --batch_size 16
"""

import argparse
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
                    help="Directory containing corpus.jsonl, queries.jsonl, qrels.tsv")
parser.add_argument("--tag",              default="tot")
parser.add_argument("--model",            default="lightonai/LateOn")
parser.add_argument("--batch_size",       type=int, default=32)
parser.add_argument("--top_k",            type=int, default=1000)
parser.add_argument("--k_values",         default="10,50,100,1000")
parser.add_argument("--index_dir",        default="lateon_plaid_index_congress",
                    help="Directory for the PLAID index")
parser.add_argument("--rebuild_index",    action="store_true",
                    help="Force rebuild the PLAID index even if it exists")
parser.add_argument("--device",           default=None,
                    help="Device (auto-detected if omitted)")
args = parser.parse_args()

k_values   = [int(k) for k in args.k_values.split(",")]
bench_dir  = Path(args.benchmark_dir)
model_slug = args.model.replace("/", "_").replace("-", "_")
index_name = f"{model_slug}_congress"

# ── Load benchmark ────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            out[d["_id"]] = d
    return out

print("[1/5] Loading benchmark...")
corpus  = load_jsonl(bench_dir / "corpus.jsonl")
queries = load_jsonl(bench_dir / "queries.jsonl")

# Qrels
qrels = {}
with open(bench_dir / "qrels.tsv") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("query-id") or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            qid, cid, score = parts
        elif len(parts) == 4:
            qid, _, cid, score = parts
        else:
            continue
        qrels.setdefault(qid, {})[cid] = int(score)
print(f"  Qrels: {sum(len(v) for v in qrels.values())} pairs across {len(qrels)} queries")

# Query metadata for breakdowns
query_witness = {}
query_memorability = {}
for qid, q in queries.items():
    meta = q.get("metadata", {})
    query_witness[qid] = meta.get("source_speaker", "unknown").lower()
    query_memorability[qid] = meta.get("memorability", 0)

corpus_ids   = list(corpus.keys())
corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
query_ids    = list(queries.keys())
query_texts  = [queries[qid]["text"] for qid in query_ids]

print(f"  Corpus:  {len(corpus_ids)} passages")
print(f"  Queries: {len(query_ids)}")

# ── Load model ────────────────────────────────────────────────────────────────

print(f"\n[2/5] Loading ColBERT model ({args.model})...")
model_kwargs = {"model_name_or_path": args.model}
if args.device:
    model_kargs["device"] = args.device
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
    print(f"\n[3/5] Building PLAID index ({len(corpus_ids):,} passages)...")
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

retriever = retrieve.ColBERT(index=index)
results_raw = retriever.retrieve(
    queries_embeddings=query_embeddings,
    k=args.top_k,
)

# Convert to run dict
run = {}
ranked_lists = {}
for i, qid in enumerate(query_ids):
    run[qid] = {}
    ranked_lists[qid] = []
    for hit in results_raw[i]:
        run[qid][hit["id"]] = hit["score"]
        ranked_lists[qid].append(hit["id"])

# ── Evaluate ──────────────────────────────────────────────────────────────────

print(f"\n[5/5] Evaluating...")

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
print(f"  LateOn ColBERT — Congressional Hearing ToT")
print(f"{'='*60}")
print(f"  Model:   {args.model}")
print(f"  Corpus:  {len(corpus_ids)} passages")
print(f"  Queries: {len(query_ids)}")
print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8}")
print(f"  {'-'*52}")
for m in ordered:
    if m in agg:
        vals = agg[m]
        print(f"  {m:<25} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

print(f"\n  Evaluated on {len(per_query)} queries")

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

# ── Save outputs ──────────────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""
results_path = bench_dir / f"lateon_{model_slug}{tag_suffix}_results.jsonl"
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
        "recall@1000":   round(v.get("recall_1000", 0), 4),
        "map":           round(v.get("map",          0), 4),
        "ranked_list":   ranked_lists.get(qid, []),
        "ranked":        sorted(run.get(qid, {}).items(), key=lambda x: -x[1])[:args.top_k],
    })
results_path.write_text("\n".join(json.dumps(r) for r in rows_out))

summary = {
    "model":       args.model,
    "tag":         args.tag,
    "corpus_size": len(corpus_ids),
    "n_queries":   len(per_query),
    "metrics":     {m: round(float(np.mean(agg.get(m, [0]))), 4) for m in ordered},
    "per_memorability_ndcg10": {
        str(m): round(float(np.mean(v)), 4) for m, v in sorted(mem_ndcg10.items(), reverse=True)
    },
    "per_witness_ndcg10": {
        w: round(float(np.mean(v)), 4) for w, v in witness_ndcg10.items()
    },
}
summary_path = bench_dir / f"lateon_{model_slug}{tag_suffix}_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults written:")
print(f"  {results_path}")
print(f"  {summary_path}")
