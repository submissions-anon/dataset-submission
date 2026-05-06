"""
BM25 evaluation for the Analogues (author attribution) benchmark.

Usage:
    python eval_bm25.py --corpus_dir /path/to/corpus_txts/

The corpus_dir should contain 1.txt, 2.txt, ... matching snippet IDs.
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
import bm25s
import pytrec_eval
from collections import defaultdict

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark_dir', required=True,
                    help='Directory with corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json')
parser.add_argument('--corpus_dir',    required=True,
                    help='Directory containing 1.txt, 2.txt, ...')
parser.add_argument('--gold_track',    default='corpus_track.csv',
                    help='Path to v2 gold CSV (snippet_id, author_name, post_title, post_url). '
                         'Used for per-author breakdown.')
parser.add_argument('--tag',           default='v2',
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
            d = json.loads(line)
            out[d['_id']] = d
    return out

def read_txt(sid, corpus_dir):
    return Path(corpus_dir, f'{sid}.txt').read_text(encoding='utf-8', errors='replace').strip()

print("Loading corpus and queries...")
corpus  = load_jsonl(bench_dir / 'corpus.jsonl')
queries = load_jsonl(bench_dir / 'queries.jsonl')

for sid, doc in corpus.items():
    doc['text'] = read_txt(sid, args.corpus_dir)
for sid, q in queries.items():
    q['text'] = read_txt(sid, args.corpus_dir)

# Binary qrels
qrels = {}
with open(bench_dir / 'qrels.tsv') as f:
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

# per-query exclusion (query itself + same-post siblings)
with open(bench_dir / 'per_query_excluded_ids.json') as f:
    excluded = json.load(f)
excluded = {qid: set(str(x) for x in ids) for qid, ids in excluded.items()}

# Author metadata for per-author breakdown (v2 gold CSV)
import csv
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

print(f"Queries: {len(queries)} | Corpus: {len(corpus)}")

# ── BM25 retrieval with per-query corpus exclusion ───────────────────────────

all_corpus_ids = list(corpus.keys())
all_corpus_texts = [corpus[cid]['text'] for cid in all_corpus_ids]
cid_to_idx = {cid: i for i, cid in enumerate(all_corpus_ids)}

# Tokenize full corpus once
print("Tokenizing corpus...")
tokenized_corpus = bm25s.tokenize(all_corpus_texts)

# Build BM25 index over full corpus
print("Building BM25 index...")
retriever = bm25s.BM25()
retriever.index(tokenized_corpus)

results = {}  # qid -> {cid: score}
ranked_lists = {}  # qid -> [cid, cid, ...] in rank order (for downstream rerankers)
# We need enough depth for both eval (max_k) and downstream feeds (--top_k).
retrieval_depth = max(max(k_values), args.top_k)

print(f"Retrieving top-{retrieval_depth} for each query (with exclusion)...")
for qid, q in queries.items():
    excl = excluded.get(qid, set())

    n_retrieve = min(retrieval_depth + len(excl) + 10, len(all_corpus_ids))
    tok_q = bm25s.tokenize([q['text']])
    docs, scores = retriever.retrieve(tok_q, corpus=all_corpus_ids, k=n_retrieve)

    ranked = {}
    ranked_order = []
    for cid, score in zip(docs[0], scores[0]):
        if cid not in excl:
            ranked[cid] = float(score)
            ranked_order.append(cid)
        if len(ranked) == retrieval_depth:
            break

    results[qid] = ranked
    ranked_lists[qid] = ranked_order

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

print("\n── BM25 Results ─────────────────────────────────────────────")
print(f"{'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8}")
print("-" * 52)

ordered_metrics = (
    [f'ndcg_cut_{k}' for k in k_values] +
    [f'recall_{k}'   for k in k_values] +
    ['map']
)
for metric in ordered_metrics:
    if metric in agg:
        vals = agg[metric]
        print(f"{metric:<25} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

print(f"\nEvaluated on {len(per_query)} queries")

# ── Per-author breakdown ──────────────────────────────────────────────────────

author_ndcg10 = defaultdict(list)
author_ndcg50 = defaultdict(list)

for qid, md in per_query.items():
    a = snippet_author.get(qid, "unknown")
    author_ndcg10[a].append(md.get("ndcg_cut_10", 0.0))
    author_ndcg50[a].append(md.get("ndcg_cut_50", 0.0))

if snippet_author:
    print(f"\n  {'─'*56}")
    print(f"  Per-author nDCG@10")
    print(f"  {'─'*56}")
    print(f"  {'Author':<30} {'nDCG@10':>8} {'nDCG@50':>8} {'n':>5}")
    print(f"  {'─'*56}")
    for author in sorted(author_ndcg10, key=lambda a: -np.mean(author_ndcg10[a])):
        v10 = author_ndcg10[author]
        v50 = author_ndcg50[author]
        print(f"  {author[:29]:<30} {np.mean(v10):>8.4f} {np.mean(v50):>8.4f} {len(v10):>5}")

# ── Save per-query results (matches downstream rerankers' input schema) ──────

tag_suffix = f"_{args.tag}" if args.tag else ""

results_path = bench_dir / f"bm25{tag_suffix}_results.jsonl"
rows_out = []
for qid in queries:
    v = per_query.get(qid, {})
    rows_out.append({
        "query_id":    qid,
        "author":      snippet_author.get(qid, "unknown"),
        "ndcg@10":     round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50":     round(v.get("ndcg_cut_50", 0), 4),
        "recall@10":   round(v.get("recall_10",   0), 4),
        "recall@50":   round(v.get("recall_50",   0), 4),
        "recall@100":  round(v.get("recall_100",  0), 4),
        "map":         round(v.get("map",          0), 4),
        "ranked_list": ranked_lists.get(qid, []),
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
    "per_author_ndcg10": {
        a: round(float(np.mean(v)), 4) for a, v in author_ndcg10.items()
    } if snippet_author else {},
}
summary_path = bench_dir / f"bm25{tag_suffix}_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[+] Summary           → {summary_path}")
