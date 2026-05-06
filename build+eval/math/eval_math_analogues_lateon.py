"""
eval_math_analogues_lateon.py
=============================
ColBERT (multi-vector) retrieval eval for the Math Reasoning-Analogue benchmark
using LateOn via PyLate + FastPLAID.

Query surface:
  --surface problem     → raw problem statement only (default)
  --surface framed      → problem + framing prompt

Handles per-query excluded IDs (self-exclusion).

Primary metric: NDCG@10
Also reports: MRR (any), MRR (score-2 only), Recall@10/50/100/300, Success@1/5/10

Requirements:
  pip install -U pylate pytrec_eval tqdm numpy

Usage:
  python eval_math_analogues_lateon.py
  python eval_math_analogues_lateon.py --surface framed
  python eval_math_analogues_lateon.py --sample
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm

from pylate import indexes, models, retrieve

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset")
parser.add_argument("--model",       default="lightonai/LateOn")
parser.add_argument("--surface",     default="framed",
                    choices=["problem", "framed"],
                    help="Query text: raw problem | full framed prompt")
parser.add_argument("--sample",      action="store_true",
                    help="Eval on stratified 500-query sample instead of full query set")
parser.add_argument("--batch-size",  type=int, default=32)
parser.add_argument("--top-k",       type=int, default=1000)
parser.add_argument("--index-dir",   default="lateon_plaid_index_math",
                    help="Directory for the PLAID index")
parser.add_argument("--rebuild-index", action="store_true",
                    help="Force rebuild the PLAID index even if it exists")
parser.add_argument("--device",      default=None,
                    help="Device (auto-detected if omitted)")
args = parser.parse_args()

dataset_dir    = Path(args.dataset_dir)
corpus_file    = dataset_dir / "corpus.jsonl"
queries_file   = dataset_dir / "queries.jsonl"
qrels_file     = dataset_dir / "qrels_addition.tsv"
exclusion_file = dataset_dir / "per_query_excluded_ids.json"

model_slug = args.model.replace("/", "_").replace("-", "_")
index_name = f"{model_slug}_math"

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

def load_queries(path, surface):
    query_ids, query_texts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if surface == "framed":
                text = q["text"]
            else:
                # Strip framing wrapper, keep just the problem block
                raw = q["text"]
                marker     = "Given the following mathematical problem:\n\n"
                end_marker = "\n\nFind other mathematical problems"
                if marker in raw and end_marker in raw:
                    start = raw.index(marker) + len(marker)
                    end   = raw.index(end_marker)
                    text  = raw[start:end].strip()
                else:
                    text = raw
            query_ids.append(q["_id"])
            query_texts.append(text)
    return query_ids, query_texts

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line
                continue
            qid, did, score = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"\n[*] Model     : {args.model}")
print(f"[*] Surface   : {args.surface}")
print(f"[*] Query set : {'sample (500)' if args.sample else 'full'}")

print("\n[1/5] Loading corpus...")
corpus_ids, corpus_texts = load_corpus(corpus_file)
print(f"  {len(corpus_ids):,} documents")

print("\n[2/5] Loading queries & qrels...")
qrels = load_qrels(qrels_file)
all_query_ids, all_query_texts = load_queries(queries_file, args.surface)

# Filter to queries with qrels
query_ids   = [qid for qid in all_query_ids if qid in qrels]
query_texts = [all_query_texts[i] for i, qid in enumerate(all_query_ids) if qid in qrels]
print(f"  {len(query_ids)} queries with qrels")

n_rel   = [len(v) for v in qrels.values()]
avg_rel = float(np.mean(n_rel))
print(f"  Avg relevant per query: {avg_rel:.1f}  (min={min(n_rel)}, max={max(n_rel)})")

# Load exclusions
excluded_ids = {}
if exclusion_file.exists():
    with open(exclusion_file) as f:
        excluded_ids = json.load(f)
    excluded_ids = {qid: set(ids) for qid, ids in excluded_ids.items()}
    affected = sum(1 for qid in query_ids if qid in excluded_ids)
    total_excluded = sum(len(excluded_ids.get(qid, [])) for qid in query_ids)
    print(f"  Loaded exclusions for {len(excluded_ids)} queries ({affected} in eval set, {total_excluded} docs)")

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

print("\n[5/5] Encoding queries & retrieving...")

query_embeddings = model.encode(
    query_texts,
    batch_size=args.batch_size,
    is_query=True,
    show_progress_bar=True,
)

# Retrieve extra to compensate for exclusion filtering
max_excluded = max((len(excluded_ids.get(qid, [])) for qid in query_ids), default=0)
retrieve_k   = min(args.top_k + max_excluded + 10, len(corpus_ids))

retriever = retrieve.ColBERT(index=index)
results_raw = retriever.retrieve(
    queries_embeddings=query_embeddings,
    k=retrieve_k,
)

# Convert to run dict, applying per-query exclusions
run = {}
for i, qid in enumerate(query_ids):
    query_excluded = excluded_ids.get(qid, set())
    docs = {}
    for hit in results_raw[i]:
        did = hit["id"]
        if did in query_excluded:
            continue
        docs[did] = hit["score"]
        if len(docs) >= args.top_k:
            break
    run[qid] = docs

# ── Evaluate ──────────────────────────────────────────────────────────────────

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {
        "ndcg_cut.10,50,100",
        "recall.10,50,100,300",
        "recip_rank",
        "success.1,5,10",
    }
)
results = evaluator.evaluate(run)

def mean(key):
    return float(np.mean([v.get(key, 0.0) for v in results.values()]))

# Best rank of any relevant doc
best_ranks = []
for qid in query_ids:
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    if ranks:
        best_ranks.append(min(ranks))

best_ranks_arr = np.array(best_ranks) if best_ranks else np.array([])

# MRR restricted to score-2 (same meta-program)
mrr_score2 = []
for qid in query_ids:
    gold2 = {did for did, s in qrels[qid].items() if s == 2}
    if not gold2:
        continue
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    for rank, (doc, _) in enumerate(ranked, 1):
        if doc in gold2:
            mrr_score2.append(1.0 / rank)
            break
    else:
        mrr_score2.append(0.0)

# ── Print ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*62}")
print(f"  {args.model}  (ColBERT / MaxSim)")
print(f"  Surface: {args.surface}  |  Queries: {len(query_ids)}")
print(f"  Avg relevant/query: {avg_rel:.1f}")
print(f"{'='*62}")
print(f"  {'Metric':<26} {'Score':>8}")
print(f"  {'-'*36}")
print(f"  {'MRR (any relevant)':<26} {mean('recip_rank'):>8.4f}")
print(f"  {'MRR (score-2 only)':<26} {float(np.mean(mrr_score2)) if mrr_score2 else 0:>8.4f}")
print(f"  {'NDCG@10':<26} {mean('ndcg_cut_10'):>8.4f}  ← primary")
print(f"  {'NDCG@50':<26} {mean('ndcg_cut_50'):>8.4f}")
print(f"  {'NDCG@100':<26} {mean('ndcg_cut_100'):>8.4f}")
print(f"  {'-'*36}")
print(f"  {'Recall@10':<26} {mean('recall_10'):>8.4f}")
print(f"  {'Recall@50':<26} {mean('recall_50'):>8.4f}")
print(f"  {'Recall@100':<26} {mean('recall_100'):>8.4f}")
print(f"  {'Recall@300':<26} {mean('recall_300'):>8.4f}")
print(f"  {'-'*36}")
print(f"  {'Success@1':<26} {mean('success_1'):>8.4f}")
print(f"  {'Success@5':<26} {mean('success_5'):>8.4f}")
print(f"  {'Success@10':<26} {mean('success_10'):>8.4f}")
if len(best_ranks_arr):
    print(f"  {'-'*36}")
    print(f"  Best-relevant-doc rank distribution:")
    for thresh in [1, 5, 10, 50, 100]:
        n = int((best_ranks_arr <= thresh).sum())
        print(f"    Top-{thresh:<5} {n:>4} / {len(query_ids)}  ({n/len(query_ids):.1%})")
print(f"{'='*62}")

# ── Save per-query results ────────────────────────────────────────────────────

suffix   = f"lateon_{model_slug}_{args.surface}"
suffix  += "_sample" if args.sample else "_full"
out_path = dataset_dir / f"results_{suffix}.jsonl"

rows = []
for qid in query_ids:
    v        = results.get(qid, {})
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    gold2    = {did for did, s in qrels[qid].items() if s == 2}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    rows.append({
        "query_id":    qid,
        "best_rank":   min(ranks) if ranks else None,
        "n_relevant":  len(rel_docs),
        "n_score2":    len(gold2),
        "mrr":         round(v.get("recip_rank",   0), 4),
        "ndcg@10":     round(v.get("ndcg_cut_10",  0), 4),
        "ndcg@50":     round(v.get("ndcg_cut_50",  0), 4),
        "ndcg@100":    round(v.get("ndcg_cut_100", 0), 4),
        "recall@10":   round(v.get("recall_10",    0), 4),
        "recall@50":   round(v.get("recall_50",    0), 4),
        "recall@100":  round(v.get("recall_100",   0), 4),
        "recall@300":  round(v.get("recall_300",   0), 4),
        "success@1":   round(v.get("success_1",    0), 4),
        "success@5":   round(v.get("success_5",    0), 4),
        "success@10":  round(v.get("success_10",   0), 4),
        "ranked":      ranked[:args.top_k],
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Per-query results → {out_path}"):
