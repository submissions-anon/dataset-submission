"""
eval_twitter_lateon.py
======================
ColBERT (multi-vector) retrieval eval for the Twitter Descriptive-IR benchmark
using LateOn via PyLate + FastPLAID.

Corpus modes:
  --corpus implicit   → 7,918 implicit tweets only
  --corpus full       → 72,122 all tweets (default)

Primary metric: NDCG@10
Recall thresholds: @10, @50, @100

Requirements:
  pip install -U pylate pytrec_eval tqdm numpy

Usage:
  python eval_twitter_lateon.py --corpus full
  python eval_twitter_lateon.py --corpus implicit
  python eval_twitter_lateon.py --corpus full --batch-size 64
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm

from pylate import indexes, models, retrieve

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset")
parser.add_argument("--model",       default="lightonai/LateOn")
parser.add_argument("--corpus",      default="full", choices=["implicit", "full"])
parser.add_argument("--batch-size",  type=int, default=32)
parser.add_argument("--top-k",       type=int, default=1000)
parser.add_argument("--index-dir",   default="lateon_plaid_index",
                    help="Directory for the PLAID index")
parser.add_argument("--rebuild-index", action="store_true",
                    help="Force rebuild the PLAID index even if it exists")
parser.add_argument("--device",      default=None,
                    help="Device (auto-detected if omitted)")
args = parser.parse_args()

dataset_dir  = Path(args.dataset_dir)
corpus_file  = dataset_dir / f"corpus_{args.corpus}.jsonl"
queries_file = dataset_dir / "queries.jsonl"
qrels_file   = dataset_dir / "qrels_addition.tsv"

model_slug   = args.model.replace("/", "_").replace("-", "_")
index_name   = f"{model_slug}_{args.corpus}"

# ── Load data ─────────────────────────────────────────────────────────────────

def load_corpus(path):
    doc_ids, doc_texts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_ids.append(doc["_id"])
            doc_texts.append(doc["text"])
    return doc_ids, doc_texts

def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, tid, score = line.split("\t")
            qrels.setdefault(qid, {})[tid] = int(score)
    return qrels

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"\n[*] Model    : {args.model}")
print(f"[*] Corpus   : {args.corpus}  ({corpus_file})")
print(f"[*] Top-k    : {args.top_k}")

print("\n[1/5] Loading corpus...")
corpus_ids, corpus_texts = load_corpus(corpus_file)
print(f"  {len(corpus_ids):,} documents")

print("\n[2/5] Loading queries & qrels...")
queries  = load_queries(queries_file)
qrels    = load_qrels(qrels_file)
query_ids   = [qid for qid in queries if qid in qrels]
query_texts = [queries[qid] for qid in query_ids]
print(f"  {len(query_ids)} queries with qrels")

avg_rel = np.mean([len(v) for v in qrels.values()])
print(f"  Avg relevant per query: {avg_rel:.1f}")

# ── Load model ────────────────────────────────────────────────────────────────

print("\n[3/5] Loading ColBERT model...")
model_kwargs = {"model_name_or_path": args.model}
if args.device:
    model_kwargs["device"] = args.device
model = models.ColBERT(**model_kwargs)

# ── Build or load PLAID index ─────────────────────────────────────────────────

index_folder = Path(args.index_dir)
index_exists = (index_folder / index_name).exists() and not args.rebuild_index

if index_exists:
    print(f"\n[4/5] Loading existing PLAID index from {index_folder / index_name}...")
    index = indexes.PLAID(
        index_folder=str(index_folder),
        index_name=index_name,
    )
else:
    print(f"\n[4/5] Building PLAID index ({len(corpus_ids):,} docs)...")
    index = indexes.PLAID(
        index_folder=str(index_folder),
        index_name=index_name,
        override=True,
    )

    # Encode all documents in one call
    print("  Encoding documents...")
    doc_embeddings = model.encode(
        corpus_texts,
        batch_size=args.batch_size,
        is_query=False,
        show_progress_bar=True,
    )

    # Add all documents to the index in one call
    print("  Adding documents to index...")
    index.add_documents(
        documents_ids=corpus_ids,
        documents_embeddings=doc_embeddings,
    )
    print(f"  Index saved to {index_folder / index_name}")

# ── Retrieve ──────────────────────────────────────────────────────────────────

print("\n[5/5] Encoding queries & retrieving...")

# Encode all queries
query_embeddings = model.encode(
    query_texts,
    batch_size=args.batch_size,
    is_query=True,
    show_progress_bar=True,
)

# Retrieve via ColBERT MaxSim
retriever = retrieve.ColBERT(index=index)
results_raw = retriever.retrieve(
    queries_embeddings=query_embeddings,
    k=args.top_k,
)

# Convert to run dict: {qid: {doc_id: score}}
run = {}
for i, qid in enumerate(query_ids):
    run[qid] = {}
    for hit in results_raw[i]:
        run[qid][hit["id"]] = hit["score"]

# ── Evaluate ──────────────────────────────────────────────────────────────────

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {
        "ndcg_cut.10,50,100",
        "recall.10,50,100",
        "recip_rank",
        "success.1,5,10",
    }
)
results = evaluator.evaluate(run)

def mean(key):
    return float(np.mean([v.get(key, 0.0) for v in results.values()]))

# Per-query rank of best relevant doc
best_ranks = []
for qid in query_ids:
    rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    if ranks:
        best_ranks.append(min(ranks))

best_ranks_arr = np.array(best_ranks)

print(f"\n{'='*62}")
print(f"  {args.model}")
print(f"  Corpus: {args.corpus} ({len(corpus_ids):,} docs)")
print(f"  Queries: {len(query_ids)}  |  Avg relevant: {avg_rel:.1f}")
print(f"{'='*62}")
print(f"  {'Metric':<22} {'Score':>8}")
print(f"  {'-'*32}")
print(f"  {'MRR':<22} {mean('recip_rank'):>8.4f}")
print(f"  {'NDCG@10':<22} {mean('ndcg_cut_10'):>8.4f}  ← primary")
print(f"  {'NDCG@50':<22} {mean('ndcg_cut_50'):>8.4f}")
print(f"  {'NDCG@100':<22} {mean('ndcg_cut_100'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Recall@10':<22} {mean('recall_10'):>8.4f}  (ceiling ~{10/avg_rel:.0%})")
print(f"  {'Recall@50':<22} {mean('recall_50'):>8.4f}  (ceiling ~100%)")
print(f"  {'Recall@100':<22} {mean('recall_100'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Success@1':<22} {mean('success_1'):>8.4f}")
print(f"  {'Success@5':<22} {mean('success_5'):>8.4f}")
print(f"  {'Success@10':<22} {mean('success_10'):>8.4f}")
print(f"  {'-'*32}")
print(f"  Best-relevant-doc rank distribution:")
for thresh in [1, 5, 10, 50, 100]:
    n = int((best_ranks_arr <= thresh).sum())
    print(f"    Top-{thresh:<5} {n:>4} / {len(query_ids)}  ({n/len(query_ids):.1%})")
print(f"{'='*62}")

# ── Save per-query results ────────────────────────────────────────────────────

out_name = f"results_lateon_{model_slug}_{args.corpus}_full.jsonl"
out_path = dataset_dir / out_name
rows = []
for qid in query_ids:
    v = results.get(qid, {})
    rel_docs  = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked    = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map  = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks     = [rank_map[d] for d in rel_docs if d in rank_map]
    best_rank = min(ranks) if ranks else None
    rows.append({
        "query_id":    qid,
        "best_rank":   best_rank,
        "n_relevant":  len(qrels[qid]),
        "mrr":         round(v.get("recip_rank",   0), 4),
        "ndcg@10":     round(v.get("ndcg_cut_10",  0), 4),
        "ndcg@50":     round(v.get("ndcg_cut_50",  0), 4),
        "ndcg@100":    round(v.get("ndcg_cut_100", 0), 4),
        "recall@10":   round(v.get("recall_10",    0), 4),
        "recall@50":   round(v.get("recall_50",    0), 4),
        "recall@100":  round(v.get("recall_100",   0), 4),
        "success@1":   round(v.get("success_1",    0), 4),
        "success@5":   round(v.get("success_5",    0), 4),
        "success@10":  round(v.get("success_10",   0), 4),
        "ranked":      sorted(run[qid].items(), key=lambda x: -x[1])[:args.top_k],
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Per-query results → {out_path}")
