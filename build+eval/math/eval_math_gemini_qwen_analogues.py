"""
eval_math_analogues.py
======================
Dense retrieval eval for the Math Reasoning-Analogue benchmark.

Supports:
  - Gemini Embedding 2  (--provider gemini)
  - Qwen3-Embedding     (--provider qwen3, local sentence-transformers)

Query surface:
  --surface problem     → raw problem statement only (what a student types)
  --surface framed      → problem + "find same reasoning pattern" framing prompt (default)
  --surface fingerprint → fingerprint_summary from corpus metadata (oracle upper bound)

Primary metric: NDCG@10  (graded: score-2 = same meta-program)
Also reports: MRR, Recall@10/50/100, Success@1/5/10

Requirements:
  pip install google-genai sentence-transformers pytrec_eval tqdm numpy

Usage:
  # Gemini, default (framed queries, full corpus)
  export GEMINI_API_KEY=...
  python eval_math_analogues.py --provider gemini

  # Qwen3 0.6B
  python eval_math_analogues.py --provider qwen3 --model Qwen/Qwen3-Embedding-0.6B

  # Qwen3 4B
  python eval_math_analogues.py --provider qwen3 --model Qwen/Qwen3-Embedding-4B

  # Oracle upper bound (embed fingerprint_summary)
  python eval_math_analogues.py --provider gemini --surface fingerprint

  # Eval on the stratified 500-query sample only
  python eval_math_analogues.py --provider gemini --sample
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
                    help="Model override (default:gemini-embedding-2-preview or "
                         "Qwen/Qwen3-Embedding-0.6B)")
parser.add_argument("--dim",         type=int, default=3072,
                    help="Output dimensionality for Gemini (3072/1536/768)")
parser.add_argument("--surface",     default="framed",
                    choices=["problem", "framed", "fingerprint"],
                    help="Query text: raw problem | framed prompt | fingerprint_summary (oracle)")
parser.add_argument("--sample",      action="store_true",
                    help="Eval on stratified 500-query sample instead of full query set")
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
corpus_file  = dataset_dir / "corpus.jsonl"
queries_file = dataset_dir / ("queries_final.jsonl")
qrels_file   = dataset_dir / ("qrels_final.tsv")

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

def load_corpus(path):
    """Returns (doc_ids, doc_texts, metadata_by_id).
    doc_texts = raw problem statement (corpus side is always the problem text).
    """
    doc_ids, doc_texts, meta = [], [], {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_ids.append(doc["_id"])
            doc_texts.append(doc["text"])
            meta[doc["_id"]] = doc.get("metadata", {})
    return doc_ids, doc_texts, meta


def load_queries(path, corpus_meta, surface):
    """Returns (query_ids, query_texts).
    surface controls what text is used as the query:
      framed      → full query_text from queries.jsonl (problem + framing prompt)
      problem     → problem statement only (strip framing wrapper)
      fingerprint → fingerprint_summary from corpus metadata (oracle)
    """
    query_ids, query_texts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qid = q["_id"]

            if surface == "framed":
                text = q["text"]
            elif surface == "problem":
                # Strip the framing wrapper — extract just the problem block
                raw = q["text"]
                marker = "Given the following mathematical problem:\n\n"
                end_marker = "\n\nFind other mathematical problems"
                if marker in raw and end_marker in raw:
                    start = raw.index(marker) + len(marker)
                    end   = raw.index(end_marker)
                    text  = raw[start:end].strip()
                else:
                    text = raw   # fallback
            elif surface == "fingerprint":
                pid  = q["metadata"].get("problem_id", qid)
                text = corpus_meta.get(pid, {}).get("fingerprint_summary", "")
                if not text:
                    text = q["text"]   # fallback if missing

            query_ids.append(qid)
            query_texts.append(text)

    return query_ids, query_texts


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)   # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, did, score = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_gemini(texts, task_type, ckpt_prefix):
    ckpt_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}_{args.surface}.npy"
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
    ckpt_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}_{args.surface}.npy"
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

print(f"\n[*] Provider  : {args.provider}")
print(f"[*] Model     : {args.model}")
print(f"[*] Surface   : {args.surface}")
print(f"[*] Query set : {'sample (500)' if args.sample else 'full'}")

print("\n[1/4] Loading corpus...")
corpus_ids, corpus_texts, corpus_meta = load_corpus(corpus_file)
print(f"  {len(corpus_ids):,} documents")

print("\n[2/4] Loading queries & qrels...")
qrels      = load_qrels(qrels_file)
query_ids, query_texts = load_queries(queries_file, corpus_meta, args.surface)
# Only eval queries that have at least one judged relevant doc
query_ids_eval   = [qid for qid in query_ids if qid in qrels]
query_texts_eval = [query_texts[i] for i, qid in enumerate(query_ids)
                    if qid in qrels]
print(f"  {len(query_ids_eval)} queries with qrels  "
      f"({len(query_ids) - len(query_ids_eval)} skipped — no qrels)")

n_rel_per_q = [len(v) for v in qrels.values()]
avg_rel = float(np.mean(n_rel_per_q))
print(f"  Avg relevant per query: {avg_rel:.1f}  "
      f"(min={min(n_rel_per_q)}, max={max(n_rel_per_q)})")

print("\n[3/4] Encoding...")
print("  Corpus:")
corp_embs = embed(corpus_texts, "RETRIEVAL_DOCUMENT", "corpus", is_query=False)
print("  Queries:")
query_embs = embed(query_texts_eval, "RETRIEVAL_QUERY", "queries", is_query=True)

# Normalize for cosine similarity
def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)

corp_embs  = normalize(corp_embs)
query_embs = normalize(query_embs)

print("\n[4/4] Retrieving & evaluating...")
SCORE_BATCH = 256
run = {}
for i in tqdm(range(0, len(query_ids_eval), SCORE_BATCH), desc="Scoring"):
    q_batch   = query_embs[i:i+SCORE_BATCH]
    scores_np = q_batch @ corp_embs.T
    for j, qid in enumerate(query_ids_eval[i:i+SCORE_BATCH]):
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

# Best-rank of any relevant doc (score >= 1) per query
best_ranks = []
for qid in query_ids_eval:
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    if ranks:
        best_ranks.append(min(ranks))

best_ranks_arr = np.array(best_ranks) if best_ranks else np.array([])

# Score-2 only MRR (same meta-program, strictest eval)
mrr_score2 = []
for qid in query_ids_eval:
    gold = {did for did, s in qrels[qid].items() if s == 2}
    if not gold:
        continue
    ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    for rank, (doc, _) in enumerate(ranked, 1):
        if doc in gold:
            mrr_score2.append(1.0 / rank)
            break
    else:
        mrr_score2.append(0.0)

suffix = f"{args.provider}_{model_slug}_{args.surface}"
suffix += "_sample" if args.sample else "_full"

# ── Build summary string (written to .txt AND printed) ────────────────────────

mrr_s2 = float(np.mean(mrr_score2)) if mrr_score2 else 0.0
lines = []
lines.append("=" * 62)
lines.append(f"  {args.model}")
lines.append(f"  Surface: {args.surface}  |  Queries: {len(query_ids_eval)}")
lines.append(f"  Avg relevant/query: {avg_rel:.1f}")
lines.append("=" * 62)
lines.append(f"  {'Metric':<26} {'Score':>8}")
lines.append(f"  {'-'*36}")
lines.append(f"  {'MRR (any relevant)':<26} {mean('recip_rank'):>8.4f}")
lines.append(f"  {'MRR (score-2 only)':<26} {mrr_s2:>8.4f}")
lines.append(f"  {'NDCG@10':<26} {mean('ndcg_cut_10'):>8.4f}  <- primary")
lines.append(f"  {'NDCG@50':<26} {mean('ndcg_cut_50'):>8.4f}")
lines.append(f"  {'NDCG@100':<26} {mean('ndcg_cut_100'):>8.4f}")
lines.append(f"  {'-'*36}")
lines.append(f"  {'Recall@10':<26} {mean('recall_10'):>8.4f}")
lines.append(f"  {'Recall@50':<26} {mean('recall_50'):>8.4f}")
lines.append(f"  {'Recall@100':<26} {mean('recall_100'):>8.4f}")
lines.append(f"  {'-'*36}")
lines.append(f"  {'Success@1':<26} {mean('success_1'):>8.4f}")
lines.append(f"  {'Success@5':<26} {mean('success_5'):>8.4f}")
lines.append(f"  {'Success@10':<26} {mean('success_10'):>8.4f}")
if len(best_ranks_arr):
    lines.append(f"  {'-'*36}")
    lines.append("  Best-relevant-doc rank distribution:")
    for thresh in [1, 5, 10, 50, 100]:
        n = int((best_ranks_arr <= thresh).sum())
        lines.append(f"    Top-{thresh:<5} {n:>4} / {len(query_ids_eval)}  ({n/len(query_ids_eval):.1%})")
lines.append("=" * 62)

summary = "\n".join(lines)
print("\n" + summary)

# Write to txt file — immune to terminal tqdm corruption
txt_path = dataset_dir / f"results_{suffix}_summary.txt"
txt_path.write_text(summary + "\n")
print(f"\n[+] Summary written to {txt_path}")

# ── Save per-query results ────────────────────────────────────────────────────

out_path = dataset_dir / f"results_{suffix}.jsonl"

rows = []
for qid in query_ids_eval:
    v        = results.get(qid, {})
    rel_docs = {did for did, s in qrels[qid].items() if s >= 1}
    gold2    = {did for did, s in qrels[qid].items() if s == 2}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r + 1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    rows.append({
        "query_id":       qid,
        "best_rank":      min(ranks) if ranks else None,
        "n_relevant":     len(rel_docs),
        "n_score2":       len(gold2),
        "mrr":            round(v.get("recip_rank",   0), 4),
        "ndcg@10":        round(v.get("ndcg_cut_10",  0), 4),
        "ndcg@50":        round(v.get("ndcg_cut_50",  0), 4),
        "ndcg@100":       round(v.get("ndcg_cut_100", 0), 4),
        "recall@10":      round(v.get("recall_10",    0), 4),
        "recall@50":      round(v.get("recall_50",    0), 4),
        "recall@100":     round(v.get("recall_100",   0), 4),
        "success@1":      round(v.get("success_1",    0), 4),
        "success@5":      round(v.get("success_5",    0), 4),
        "success@10":     round(v.get("success_10",   0), 4),
        "ranked":         ranked,  # full top-k list for NDCG-Pooled
    })

print(f"\n[+] Saving {len(rows)} per-query results → {out_path}")
out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"    Done.")
