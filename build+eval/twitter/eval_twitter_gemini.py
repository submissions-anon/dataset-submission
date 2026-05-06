"""
eval_twitter.py
===============
Dense retrieval eval for the Twitter Descriptive-IR benchmark.

Supports:
  - Gemini Embedding 2  (provider=gemini)
  - Qwen3-Embedding     (provider=qwen3, local sentence-transformers)

Corpus modes:
  --corpus implicit   → 7,918 implicit tweets only (default)
  --corpus full       → 72,122 all tweets

Embed surface:
  --surface tweet     → raw tweet text (standard eval)
  --surface desc      → GPT-5 implicit_meaning descriptions (oracle upper bound)

Primary metric: NDCG@10
Recall thresholds: @10, @50, @100  (avg relevant = ~21.4, so @50 ~ 100% ceiling)

Requirements:
  pip install google-genai sentence-transformers pytrec_eval tqdm numpy

Usage:
  # Gemini
  export GEMINI_API_KEY=...
  python eval_twitter.py --provider gemini --corpus implicit --surface tweet

  # Qwen3
  python eval_twitter.py --provider qwen3 --model Qwen/Qwen3-Embedding-0.6B --corpus implicit

  # Oracle upper bound (embed descriptions)
  python eval_twitter.py --provider gemini --surface desc
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset")
parser.add_argument("--provider",    default="gemini", choices=["gemini", "qwen3"])
parser.add_argument("--model",       default=None,
                    help="Model name (default: gemini-embedding-2-preview or Qwen/Qwen3-Embedding-0.6B)")
parser.add_argument("--dim",         type=int, default=3072,
                    help="Output dimensionality for Gemini (3072/1536/768)")
parser.add_argument("--corpus",      default="full", choices=["implicit", "full"],
                    help="Which corpus to eval against")
parser.add_argument("--surface",     default="tweet", choices=["tweet", "desc"],
                    help="tweet=raw text, desc=implicit_meaning descriptions (oracle)")
parser.add_argument("--batch-size",  type=int, default=50)
parser.add_argument("--top-k",       type=int, default=1000)
parser.add_argument("--sleep",       type=float, default=1.0)
parser.add_argument("--ckpt-dir",    default=None)
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)

if args.model is None:
    args.model = ("gemini-embedding-2-preview" if args.provider == "gemini"
                  else "Qwen/Qwen3-Embedding-0.6B")

ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(f"eval_cache_{args.provider}")
ckpt_dir.mkdir(exist_ok=True)

model_slug   = args.model.replace("/", "_").replace("-", "_")
corpus_file  = dataset_dir / f"corpus_{args.corpus}.jsonl"
queries_file = dataset_dir / "queries_merged.jsonl" #"queries_rewritten.jsonl"
qrels_file   = dataset_dir / "qrels_merged.tsv"

# ── Provider setup ────────────────────────────────────────────────────────────

if args.provider == "gemini":
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        raise ImportError("pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY")
    gclient = genai.Client(api_key=api_key)

elif args.provider == "qwen3":
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")
    print(f"Loading {args.model}...")
    st_model = SentenceTransformer(args.model, trust_remote_code=True)

# ── Load data ─────────────────────────────────────────────────────────────────

def load_corpus(path, surface):
    doc_ids, doc_texts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            tid = doc["_id"]
            if surface == "desc":
                text = doc.get("metadata", {}).get("implicit_meaning", "").strip()
                if not text:
                    text = doc["text"]  # fallback to raw if no description
            else:
                text = doc["text"]
            doc_ids.append(tid)
            doc_texts.append(text)
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

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_gemini(texts, task_type, ckpt_prefix):
    ckpt_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}_{args.corpus}_{args.surface}.npy"
    if ckpt_path.exists():
        print(f"  Loaded from cache: {ckpt_path}")
        return np.load(ckpt_path)

    all_embs = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc=f"  Embedding {ckpt_prefix}"):
        batch = texts[i:i+args.batch_size]
        for attempt in range(5):
            try:
                cfg = gtypes.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=args.dim,
                )
                result = gclient.models.embed_content(
                    model=args.model,
                    contents=batch,
                    config=cfg,
                )
                vecs = np.array([e.values for e in result.embeddings], dtype=np.float32)
                all_embs.append(vecs)
                time.sleep(args.sleep)
                break
            except Exception as e:
                wait = 30 * (2 ** attempt)
                print(f"\n  [!] attempt {attempt+1}/5: {e} — retry in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding failed after 5 attempts at batch {i}")

    embs = np.concatenate(all_embs, axis=0)
    np.save(ckpt_path, embs)
    print(f"  Saved {ckpt_path}")
    return embs

def embed_qwen3(texts, ckpt_prefix, is_query=False):
    ckpt_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}_{args.corpus}_{args.surface}.npy"
    if ckpt_path.exists():
        print(f"  Loaded from cache: {ckpt_path}")
        return np.load(ckpt_path)

    prompt_name = "query" if is_query else None
    embs = st_model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        prompt_name=prompt_name,
        normalize_embeddings=True,
    )
    embs = embs.astype(np.float32)
    np.save(ckpt_path, embs)
    print(f"  Saved {ckpt_path}")
    return embs

def embed(texts, task_type_gemini, ckpt_prefix, is_query=False):
    if args.provider == "gemini":
        return embed_gemini(texts, task_type_gemini, ckpt_prefix)
    else:
        return embed_qwen3(texts, ckpt_prefix, is_query=is_query)

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"\n[*] Provider : {args.provider}")
print(f"[*] Model    : {args.model}")
print(f"[*] Corpus   : {args.corpus}  ({corpus_file})")
print(f"[*] Surface  : {args.surface}")

print("\n[1/4] Loading corpus...")
corpus_ids, corpus_texts = load_corpus(corpus_file, args.surface)
print(f"  {len(corpus_ids):,} documents")

print("\n[2/4] Loading queries & qrels...")
queries  = load_queries(queries_file)
qrels    = load_qrels(qrels_file)
# Only eval queries present in both queries and qrels
query_ids   = [qid for qid in queries if qid in qrels]
query_texts = [queries[qid] for qid in query_ids]
print(f"  {len(query_ids)} queries with qrels")

# Avg relevant per query
avg_rel = np.mean([len(v) for v in qrels.values()])
print(f"  Avg relevant per query: {avg_rel:.1f}")

print("\n[3/4] Encoding...")
print("  Corpus:")
corp_embs = embed(corpus_texts, "RETRIEVAL_DOCUMENT", "corpus", is_query=False)

print("  Queries:")
query_embs = embed(query_texts, "RETRIEVAL_QUERY", "queries", is_query=True)

# Normalize for cosine sim
def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)

corp_embs  = normalize(corp_embs)
query_embs = normalize(query_embs)

print("\n[4/4] Retrieving & evaluating...")
SCORE_BATCH = 256
run = {}
for i in tqdm(range(0, len(query_ids), SCORE_BATCH), desc="Scoring"):
    q_batch   = query_embs[i:i+SCORE_BATCH]
    scores_np = q_batch @ corp_embs.T
    for j, qid in enumerate(query_ids[i:i+SCORE_BATCH]):
        row     = scores_np[j]
        top_idx = np.argsort(row)[::-1][:args.top_k]
        run[qid] = {corpus_ids[k]: float(row[k]) for k in top_idx}

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

# Per-query rank of best relevant doc (score=2 preferred, else score=1)
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
print(f"  Corpus: {args.corpus} ({len(corpus_ids):,} docs)  |  Surface: {args.surface}")
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

# ── Save ──────────────────────────────────────────────────────────────────────

out_name = f"results_{args.provider}_{model_slug}_{args.corpus}_{args.surface}_full_365q.jsonl"
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
        "ranked":       sorted(run[qid].items(), key=lambda x: -x[1])[:args.top_k],
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Per-query results → {out_path}")
