"""
BM25 evaluation for the Congressional Hearing ToT benchmark.

Usage:
    python eval_bm25_congress.py --benchmark_dir congress_corpus_data/beir_export/

Expects:
    benchmark_dir/corpus.jsonl   - {"_id": ..., "title": ..., "text": ..., "metadata": {...}}
    benchmark_dir/queries.jsonl  - {"_id": ..., "text": ..., "metadata": {...}}
    benchmark_dir/qrels.tsv      - query-id  corpus-id  score
"""

import json
import argparse
from pathlib import Path
import numpy as np
import bm25s
import pytrec_eval
from collections import defaultdict

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark_dir', required=True,
                    help='Directory with corpus.jsonl, queries.jsonl, qrels.tsv')
parser.add_argument('--tag',           default='tot',
                    help='Suffix for output filenames')
parser.add_argument('--k_values',      default='10,50,100,1000',
                    help='Comma-separated cutoffs for nDCG/Recall')
parser.add_argument('--top_k',         type=int, default=1000,
                    help='How many ranked docs to save per query (for downstream rerankers)')
args = parser.parse_args()

k_values = [int(k) for k in args.k_values.split(',')]
bench_dir = Path(args.benchmark_dir)

# ── load benchmark ────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            d = json.loads(line)
            out[d['_id']] = d
    return out

print("Loading corpus...")
corpus = load_jsonl(bench_dir / 'corpus.jsonl')
print(f"  Corpus: {len(corpus)} passages")

print("Loading queries...")
queries = load_jsonl(bench_dir / 'queries.jsonl')
print(f"  Queries: {len(queries)}")

# Qrels
qrels = {}
with open(bench_dir / 'qrels.tsv') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('query-id') or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) == 3:
            qid, cid, score = parts
        elif len(parts) == 4:
            # TREC format: query-id Q0 corpus-id score
            qid, _, cid, score = parts
        else:
            continue
        qrels.setdefault(qid, {})[cid] = int(score)

print(f"  Qrels: {sum(len(v) for v in qrels.values())} pairs across {len(qrels)} queries")

# Build query metadata index for per-witness breakdown
query_witness = {}   # qid -> witness name (from source passage speaker)
query_memorability = {}  # qid -> memorability score
for qid, q in queries.items():
    meta = q.get('metadata', {})
    speaker = meta.get('source_speaker', '').lower()
    query_witness[qid] = speaker
    query_memorability[qid] = meta.get('memorability', 0)

# ── BM25 retrieval ───────────────────────────────────────────────────────────

all_corpus_ids = list(corpus.keys())
all_corpus_texts = [corpus[cid]['text'] for cid in all_corpus_ids]

print("Tokenizing corpus...")
tokenized_corpus = bm25s.tokenize(all_corpus_texts)

print("Building BM25 index...")
retriever = bm25s.BM25()
retriever.index(tokenized_corpus)

retrieval_depth = max(max(k_values), args.top_k)

results = {}        # qid -> {cid: score}
ranked_lists = {}   # qid -> [cid, ...] in rank order

print(f"Retrieving top-{retrieval_depth} for each query...")
for i, (qid, q) in enumerate(queries.items()):
    n_retrieve = min(retrieval_depth + 10, len(all_corpus_ids))
    tok_q = bm25s.tokenize([q['text']])
    docs, scores = retriever.retrieve(tok_q, corpus=all_corpus_ids, k=n_retrieve)

    ranked = {}
    ranked_order = []
    for cid, score in zip(docs[0], scores[0]):
        ranked[cid] = float(score)
        ranked_order.append(cid)
        if len(ranked) == retrieval_depth:
            break

    results[qid] = ranked
    ranked_lists[qid] = ranked_order

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(queries)} queries done")

# ── pytrec_eval ───────────────────────────────────────────────────────────────

metrics = set()
for k in k_values:
    metrics.add(f'ndcg_cut_{k}')
    metrics.add(f'recall_{k}')
metrics.add('map')

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
per_query = evaluator.evaluate(results)

# ── aggregate and print ───────────────────────────────────────────────────────

agg = defaultdict(list)
for qid, metric_scores in per_query.items():
    for metric, val in metric_scores.items():
        agg[metric].append(val)

ordered_metrics = (
    [f'ndcg_cut_{k}' for k in k_values] +
    [f'recall_{k}'   for k in k_values] +
    ['map']
)

print(f"\n── BM25 Results ─────────────────────────────────────────────")
print(f"{'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8}")
print("-" * 52)
for metric in ordered_metrics:
    if metric in agg:
        vals = agg[metric]
        print(f"{metric:<25} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

print(f"\nEvaluated on {len(per_query)} queries")

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

# ── Save per-query results ────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""

results_path = bench_dir / f"bm25{tag_suffix}_results.jsonl"
rows_out = []
for qid in queries:
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
    })
results_path.write_text("\n".join(json.dumps(r) for r in rows_out))
print(f"\n[+] Per-query results → {results_path}")

# ── Save summary ──────────────────────────────────────────────────────────────

summary = {
    "model":       "BM25",
    "tag":         args.tag,
    "corpus_size": len(corpus),
    "n_queries":   len(per_query),
    "metrics":     {m: round(float(np.mean(agg.get(m, [0]))), 4) for m in ordered_metrics},
    "per_memorability_ndcg10": {
        str(m): round(float(np.mean(v)), 4) for m, v in sorted(mem_ndcg10.items(), reverse=True)
    },
    "per_witness_ndcg10": {
        w: round(float(np.mean(v)), 4) for w, v in witness_ndcg10.items()
    },
}
summary_path = bench_dir / f"bm25{tag_suffix}_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[+] Summary           → {summary_path}")
