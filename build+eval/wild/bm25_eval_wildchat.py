"""
bm25_eval_wildchat.py
=====================
BM25 baseline eval for the WildChat Descriptive-IR benchmark.
Queries describe LLM failure modes; corpus is 507K conversations.

Memory-safe: corpus is streamed for tokenisation, never fully loaded as strings.
BM25 index is built once and cached to disk as a pickle.

Primary metric : NDCG@10
Recall cutoffs : @50 (ceiling 85%), @100 (ceiling 94%), @500 (ceiling 100%)

Requirements:
  pip install rank_bm25 pytrec_eval tqdm numpy

Usage:
  # Original queries
  python bm25_eval_wildchat.py --dataset-dir dataset

  # Merged queries (recommended)
  python bm25_eval_wildchat.py --dataset-dir dataset/merged

  # Skip rebuild if cache already exists
  python bm25_eval_wildchat.py --dataset-dir dataset/merged --no-rebuild
"""

import argparse, json, pickle, re, time
from pathlib import Path

import numpy as np
import pytrec_eval
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset/abstract",
                    help="Directory with queries.jsonl and qrels.tsv")
parser.add_argument("--top-k",       type=int, default=1000)
parser.add_argument("--cache-dir",   default="bm25_cache")
parser.add_argument("--no-rebuild",  action="store_true",
                    help="Use cached index; fail if not found")
args = parser.parse_args()

dataset_dir  = Path(args.dataset_dir)
cache_dir    = Path(args.cache_dir)
cache_dir.mkdir(exist_ok=True)

corpus_file  = Path("dataset") / "corpus.jsonl"   # always here
queries_file = dataset_dir / "queries.jsonl"
qrels_file   = dataset_dir / "qrels.tsv"
cache_file   = cache_dir   / "bm25_wildchat.pkl"
ids_file     = cache_dir   / "bm25_wildchat_ids.json"

for f in [corpus_file, queries_file, qrels_file]:
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

# ── Tokeniser ─────────────────────────────────────────────────────────────────

def tokenise(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]

# ── Corpus streaming ──────────────────────────────────────────────────────────

def iter_corpus(path):
    """Yield (id, text) one at a time — never all in RAM."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                yield doc["_id"], doc["text"]

# ── BM25 index ────────────────────────────────────────────────────────────────

def build_or_load_index(path, cache_path, ids_path, force_rebuild=False):
    if cache_path.exists() and ids_path.exists() and not force_rebuild:
        print(f"  Loading BM25 index from cache: {cache_path}")
        t0 = time.time()
        with open(cache_path, "rb") as f:
            bm25 = pickle.load(f)
        doc_ids = json.loads(ids_path.read_text())
        print(f"  Loaded {len(doc_ids):,} docs in {time.time()-t0:.1f}s")
        return bm25, doc_ids

    if args.no_rebuild:
        raise FileNotFoundError(f"Cache not found: {cache_path}. Remove --no-rebuild to build.")

    print(f"  Streaming and tokenising corpus...")
    doc_ids   = []
    tokenised = []
    for did, text in tqdm(iter_corpus(path), desc="  Tokenising", unit="doc"):
        doc_ids.append(did)
        tokenised.append(tokenise(text))

    print(f"  Building BM25 index over {len(doc_ids):,} docs...")
    t0   = time.time()
    bm25 = BM25Okapi(tokenised)
    print(f"  Built in {time.time()-t0:.1f}s")

    with open(cache_path, "wb") as f:
        pickle.dump(bm25, f)
    ids_path.write_text(json.dumps(doc_ids))
    print(f"  Saved index → {cache_path}")
    print(f"  Saved IDs   → {ids_path}")
    return bm25, doc_ids

# ── Query / qrel loading ──────────────────────────────────────────────────────

def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                queries[q["_id"]] = q["text"]
    return queries

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, cid, score = line.split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)
    return qrels

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"\n[*] Dataset dir : {dataset_dir}")
print(f"[*] Corpus      : {corpus_file}")
print(f"[*] Top-k       : {args.top_k}")

print("\n[1/4] Building/loading BM25 index...")
bm25, doc_ids = build_or_load_index(
    corpus_file, cache_file, ids_file,
    force_rebuild=(not args.no_rebuild and not cache_file.exists()),
)
print(f"  {len(doc_ids):,} documents indexed")

print("\n[2/4] Loading queries & qrels...")
queries  = load_queries(queries_file)
qrels    = load_qrels(qrels_file)
query_ids   = [qid for qid in queries if qid in qrels]
query_texts = [queries[qid] for qid in query_ids]
print(f"  {len(query_ids)} queries with qrels")
counts  = [len(v) for v in qrels.values()]
avg_rel = np.mean(counts)
print(f"  Avg rel/query: {avg_rel:.1f}  |  Median: {np.median(counts):.1f}  |  Min/Max: {min(counts)}/{max(counts)}")

corpus_id_set = set(doc_ids)
missing = sum(1 for qid in query_ids for cid in qrels[qid] if cid not in corpus_id_set)
if missing:
    print(f"  [!] {missing} qrel entries reference docs not in corpus")

print("\n[3/4] BM25 retrieval...")
run = {}
for qid, qtext in tqdm(zip(query_ids, query_texts), total=len(query_ids), desc="  Retrieving"):
    tokens  = tokenise(qtext)
    scores  = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:args.top_k]
    run[qid] = {doc_ids[i]: float(scores[i]) for i in top_idx}

print("\n[4/4] Evaluating...")
evaluator = pytrec_eval.RelevanceEvaluator(
    qrels,
    {
        "ndcg_cut.10,50,100",
        "recall.50,100,500",
        "recip_rank",
        "success.1,5,10",
    }
)
results = evaluator.evaluate(run)

def mean(key):
    return float(np.mean([v.get(key, 0.0) for v in results.values()]))

# Best-rank distribution
best_ranks = []
for qid in query_ids:
    rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    if ranks:
        best_ranks.append(min(ranks))
best_ranks_arr = np.array(best_ranks) if best_ranks else np.array([])

print(f"\n{'='*62}")
print(f"  BM25Okapi — WildChat Descriptive-IR")
print(f"  Corpus: {len(doc_ids):,} docs  |  Queries: {len(query_ids)}  |  Avg rel: {avg_rel:.1f}")
print(f"{'='*62}")
print(f"  {'Metric':<22} {'Score':>8}")
print(f"  {'-'*32}")
print(f"  {'MRR':<22} {mean('recip_rank'):>8.4f}")
print(f"  {'NDCG@10':<22} {mean('ndcg_cut_10'):>8.4f}  ← primary")
print(f"  {'NDCG@50':<22} {mean('ndcg_cut_50'):>8.4f}")
print(f"  {'NDCG@100':<22} {mean('ndcg_cut_100'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Recall@50':<22} {mean('recall_50'):>8.4f}  (ceiling  85%)")
print(f"  {'Recall@100':<22} {mean('recall_100'):>8.4f}  (ceiling  94%)")
print(f"  {'Recall@500':<22} {mean('recall_500'):>8.4f}  (ceiling 100%)")
print(f"  {'-'*32}")
print(f"  {'Success@1':<22} {mean('success_1'):>8.4f}")
print(f"  {'Success@5':<22} {mean('success_5'):>8.4f}")
print(f"  {'Success@10':<22} {mean('success_10'):>8.4f}")
print(f"  {'-'*32}")
if len(best_ranks_arr):
    print(f"  Best-relevant-doc rank:")
    for thresh in [1, 5, 10, 50, 100, 500]:
        n = int((best_ranks_arr <= thresh).sum())
        print(f"    Top-{thresh:<5} {n:>3}/{len(query_ids)}  ({n/len(query_ids):.0%})")
print(f"{'='*62}")

# ── Per-query breakdown ───────────────────────────────────────────────────────

print(f"\n  {'qid':<8} {'n_rel':>5}  {'NDCG@10':>8}  {'R@50':>7}  {'R@100':>7}  {'R@500':>7}  {'best_rank':>9}")
print(f"  {'-'*62}")
for qid in query_ids:
    v        = results.get(qid, {})
    rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    best     = min(ranks) if ranks else -1
    print(f"  {qid:<8} {len(qrels[qid]):>5}  "
          f"{v.get('ndcg_cut_10',0):>8.4f}  "
          f"{v.get('recall_50',0):>7.4f}  "
          f"{v.get('recall_100',0):>7.4f}  "
          f"{v.get('recall_500',0):>7.4f}  "
          f"{best:>9}")

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = dataset_dir / f"results_bm25_{dataset_dir.name}.jsonl"
rows = []
for qid in query_ids:
    v        = results.get(qid, {})
    rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    rows.append({
        "query_id":   qid,
        "query_text": queries[qid][:120],
        "best_rank":  min(ranks) if ranks else None,
        "n_relevant": len(qrels[qid]),
        "mrr":        round(v.get("recip_rank",   0), 4),
        "ndcg@10":    round(v.get("ndcg_cut_10",  0), 4),
        "ndcg@50":    round(v.get("ndcg_cut_50",  0), 4),
        "ndcg@100":   round(v.get("ndcg_cut_100", 0), 4),
        "recall@50":  round(v.get("recall_50",    0), 4),
        "recall@100": round(v.get("recall_100",   0), 4),
        "recall@500": round(v.get("recall_500",   0), 4),
        "success@1":  round(v.get("success_1",    0), 4),
        "success@5":  round(v.get("success_5",    0), 4),
        "success@10": round(v.get("success_10",   0), 4),
        "ranked":     sorted(run[qid].items(), key=lambda x: -x[1])[:args.top_k],
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Per-query results → {out_path}")
