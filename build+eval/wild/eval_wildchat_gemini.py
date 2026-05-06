"""
eval_wildchat.py
================
Dense retrieval eval for the WildChat Descriptive-IR benchmark.
Queries describe LLM failure modes; corpus is 507K conversations.

Memory-safe  : corpus texts streamed one at a time, never held in RAM.
Rate-safe    : exponential backoff on 429s, configurable sleep between batches.
Crash-safe   : incremental checkpoint every --save-every docs; resumes on restart.

Supports:
  - Gemini Embedding 2  (--provider gemini)
  - Qwen3-Embedding     (--provider qwen3, local sentence-transformers)

Primary metric : NDCG@10
Recall cutoffs : @10, @50 (ceiling 85%), @100 (ceiling 94%), @500 (ceiling 100%)

Requirements:
  pip install google-genai sentence-transformers pytrec_eval tqdm numpy

Usage:
  # Gemini, merged queries (recommended, run overnight at tier 1)
  export GEMINI_API_KEY=...
  python eval_wildchat.py --provider gemini --dataset-dir dataset/merged --sleep 4

  # Qwen3 local (no rate limits)
  python eval_wildchat.py --provider qwen3 --model Qwen/Qwen3-Embedding-0.6B --dataset-dir dataset/merged
"""

import argparse, json, os, time
from pathlib import Path
import numpy as np
import pytrec_eval
import torch
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset/abstract",
                    help="Directory with queries.jsonl and qrels.tsv")
parser.add_argument("--provider",    default="gemini", choices=["gemini", "qwen3"])
parser.add_argument("--model",       default=None)
parser.add_argument("--dim",         type=int, default=3072,
                    help="Gemini output dimensionality (3072/1536/768)")
parser.add_argument("--batch-size",  type=int, default=50,
                    help="Docs per embedding API call")
parser.add_argument("--top-k",       type=int, default=1000,
                    help="Docs to retrieve per query for evaluation")
parser.add_argument("--sleep",       type=float, default=1.0,
                    help="Seconds to sleep between embedding batches (increase for tier 1)")
parser.add_argument("--save-every",  type=int, default=5000,
                    help="Save incremental corpus checkpoint every N docs")
parser.add_argument("--ckpt-dir",    default=None,
                    help="Directory to store embedding caches (default: eval_cache_<provider>)")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
corpus_file  = Path("dataset") / "corpus.jsonl"   # always here
queries_file = dataset_dir / "queries.jsonl"
qrels_file   = dataset_dir / "qrels.tsv"

for f in [corpus_file, queries_file, qrels_file]:
    if not f.exists():
        raise FileNotFoundError(f"Missing required file: {f}")

if args.model is None:
    args.model = ("gemini-embedding-2-preview" if args.provider == "gemini"
                  else "Qwen/Qwen3-Embedding-0.6B")

ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(f"eval_cache_{args.provider}")
ckpt_dir.mkdir(exist_ok=True)
model_slug = args.model.replace("/", "_").replace("-", "_")

# ── Provider setup ────────────────────────────────────────────────────────────

if args.provider == "gemini":
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        raise ImportError("pip install google-genai")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")
    gclient = genai.Client(api_key=api_key)

elif args.provider == "qwen3":
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")
    print(f"Loading {args.model}...")
    st_model = SentenceTransformer(args.model, trust_remote_code=True)

# ── Corpus helpers ────────────────────────────────────────────────────────────

def load_corpus_ids(path):
    """Load only document IDs into RAM."""
    doc_ids = []
    with open(path) as f:
        for line in tqdm(f, desc="  Loading IDs", unit="doc"):
            line = line.strip()
            if line:
                doc_ids.append(json.loads(line)["_id"])
    return doc_ids

def iter_corpus_texts(path):
    """Yield one document text at a time — never all in RAM."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]

# ── Gemini embedding ──────────────────────────────────────────────────────────

def _gemini_embed_batch(batch, task_type):
    """Embed one batch with exponential backoff on 429s."""
    for attempt in range(6):
        try:
            cfg = gtypes.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=args.dim,
            )
            result = gclient.models.embed_content(
                model=args.model, contents=batch, config=cfg,
            )
            time.sleep(args.sleep)
            return np.array([e.values for e in result.embeddings], dtype=np.float32)
        except Exception as e:
            wait = 60 * (2 ** attempt)
            print(f"\n  [!] attempt {attempt+1}/6: {e}")
            print(f"      Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Embedding failed after 6 attempts — check quota/billing")

def embed_corpus_gemini(path, ckpt_prefix):
    """
    Stream corpus through Gemini with incremental checkpointing.
    Resumes from last checkpoint if interrupted.
    """
    final_path   = ckpt_dir / f"{ckpt_prefix}_{model_slug}.npy"
    partial_emb  = ckpt_dir / f"{ckpt_prefix}_{model_slug}_partial.npy"
    partial_n    = ckpt_dir / f"{ckpt_prefix}_{model_slug}_partial_n.txt"

    if final_path.exists():
        print(f"  Corpus embeddings loaded from cache: {final_path}")
        return np.load(final_path)

    # Resume from partial checkpoint if available
    if partial_emb.exists() and partial_n.exists():
        saved_embs = np.load(partial_emb)
        docs_done  = int(partial_n.read_text().strip())
        all_embs   = [saved_embs]
        print(f"  Resuming from checkpoint: {docs_done:,} docs already embedded")
    else:
        all_embs  = []
        docs_done = 0

    batch      = []
    total_seen = 0

    for text in tqdm(iter_corpus_texts(path), desc="  Embedding corpus", unit="doc"):
        total_seen += 1

        # Skip already-embedded docs on resume
        if total_seen <= docs_done:
            continue

        batch.append(text)

        if len(batch) >= args.batch_size:
            all_embs.append(_gemini_embed_batch(batch, "RETRIEVAL_DOCUMENT"))
            batch = []

            # Incremental checkpoint
            if (total_seen % args.save_every) < args.batch_size:
                partial = np.concatenate(all_embs, axis=0)
                np.save(partial_emb, partial)
                partial_n.write_text(str(total_seen))
                print(f"\n  [checkpoint] {total_seen:,} docs saved")

    # Flush remaining
    if batch:
        all_embs.append(_gemini_embed_batch(batch, "RETRIEVAL_DOCUMENT"))

    embs = np.concatenate(all_embs, axis=0)
    np.save(final_path, embs)
    partial_emb.unlink(missing_ok=True)
    partial_n.unlink(missing_ok=True)
    print(f"  Corpus embeddings saved: {final_path}  ({embs.shape})")
    return embs

def embed_queries_gemini(texts, ckpt_prefix):
    """Embed queries (small — straightforward batching)."""
    ckpt_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}.npy"
    if ckpt_path.exists():
        print(f"  Query embeddings loaded from cache: {ckpt_path}")
        return np.load(ckpt_path)

    all_embs = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="  Embedding queries"):
        batch = texts[i:i+args.batch_size]
        all_embs.append(_gemini_embed_batch(batch, "RETRIEVAL_QUERY"))

    embs = np.concatenate(all_embs, axis=0)
    np.save(ckpt_path, embs)
    print(f"  Query embeddings saved: {ckpt_path}")
    return embs

# ── Qwen3 embedding ───────────────────────────────────────────────────────────
def embed_corpus_qwen3(path, ckpt_prefix):
    final_path   = ckpt_dir / f"{ckpt_prefix}_{model_slug}.npy"
    partial_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}_partial.npy"
    partial_n    = ckpt_dir / f"{ckpt_prefix}_{model_slug}_partial_n.txt"

    if final_path.exists():
        print(f"  Corpus embeddings loaded from cache: {final_path}")
        return np.load(final_path)

    # Resume
    if partial_path.exists() and partial_n.exists():
        all_embs  = [np.load(partial_path)]
        docs_done = int(partial_n.read_text().strip())
        print(f"  Resuming from checkpoint: {docs_done:,} docs already embedded")
    else:
        all_embs  = []
        docs_done = 0

    CHUNK      = 10000  # keep large for speed
    chunk      = []
    total_seen = 0
    chunks_since_save = 0

    for text in tqdm(iter_corpus_texts(path), desc="  Embedding corpus", unit="doc"):
        total_seen += 1
        if total_seen <= docs_done:
            continue
        chunk.append(text)
        if len(chunk) >= CHUNK:
            embs = st_model.encode(
                chunk, batch_size=args.batch_size,
                show_progress_bar=False, normalize_embeddings=True,
            ).astype(np.float32)
            all_embs.append(embs)
            chunk = []
            torch.cuda.empty_cache()
            chunks_since_save += 1

            # Save every 5 chunks = every 50K docs
            if chunks_since_save >= 1:
                partial = np.concatenate(all_embs, axis=0)
                np.save(partial_path, partial)
                partial_n.write_text(str(partial.shape[0]))
                print(f"\n  [checkpoint] {partial.shape[0]:,} docs saved")
                chunks_since_save = 0

    if chunk:
        embs = st_model.encode(
            chunk, batch_size=args.batch_size,
            show_progress_bar=False, normalize_embeddings=True,
        ).astype(np.float32)
        all_embs.append(embs)

    embs = np.concatenate(all_embs, axis=0)
    np.save(final_path, embs)
    partial_path.unlink(missing_ok=True)
    partial_n.unlink(missing_ok=True)
    print(f"  Corpus embeddings saved: {final_path}  ({embs.shape})")
    return embs

def embed_queries_qwen3(texts, ckpt_prefix):
    ckpt_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}.npy"
    if ckpt_path.exists():
        print(f"  Query embeddings loaded from cache: {ckpt_path}")
        return np.load(ckpt_path)

    embs = st_model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True,
        prompt_name="query", normalize_embeddings=True,
    ).astype(np.float32)
    np.save(ckpt_path, embs)
    print(f"  Query embeddings saved: {ckpt_path}")
    return embs

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
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, cid, score = line.split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)
    return qrels

def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"\n{'='*62}")
print(f"  Provider   : {args.provider}")
print(f"  Model      : {args.model}")
print(f"  Queries    : {queries_file}")
print(f"  Corpus     : {corpus_file}")
print(f"  Sleep      : {args.sleep}s/batch")
print(f"  Save every : {args.save_every} docs")
print(f"{'='*62}")

print("\n[1/4] Loading corpus IDs...")
corpus_ids = load_corpus_ids(corpus_file)
print(f"  {len(corpus_ids):,} documents")

print("\n[2/4] Loading queries & qrels...")
queries  = load_queries(queries_file)
qrels    = load_qrels(qrels_file)
query_ids   = [qid for qid in queries if qid in qrels]
query_texts = [queries[qid] for qid in query_ids]
print(f"  {len(query_ids)} queries with qrels")
counts  = [len(v) for v in qrels.values()]
avg_rel = np.mean(counts)
print(f"  Avg rel/query: {avg_rel:.1f}  |  Median: {np.median(counts):.1f}  |  Min/Max: {min(counts)}/{max(counts)}")

print("\n[3/4] Encoding...")
query_ckpt = f"queries_wildchat_{dataset_dir.name}"
if args.provider == "gemini":
    corp_embs  = normalize(embed_corpus_gemini(corpus_file,   "corpus_wildchat"))
    query_embs = normalize(embed_queries_gemini(query_texts,   query_ckpt))
else:
    corp_embs  = normalize(embed_corpus_qwen3(corpus_file,    "corpus_wildchat"))
    query_embs = normalize(embed_queries_qwen3(query_texts,    query_ckpt))

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
        "recall.10,50,100,500",
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
best_ranks_arr = np.array(best_ranks)

print(f"\n{'='*62}")
print(f"  {args.model}")
print(f"  Corpus: {len(corpus_ids):,} docs  |  Queries: {len(query_ids)}  |  Avg rel: {avg_rel:.1f}")
print(f"{'='*62}")
print(f"  {'Metric':<22} {'Score':>8}")
print(f"  {'-'*32}")
print(f"  {'MRR':<22} {mean('recip_rank'):>8.4f}")
print(f"  {'NDCG@10':<22} {mean('ndcg_cut_10'):>8.4f}  ← primary")
print(f"  {'NDCG@50':<22} {mean('ndcg_cut_50'):>8.4f}")
print(f"  {'NDCG@100':<22} {mean('ndcg_cut_100'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Recall@10':<22} {mean('recall_10'):>8.4f}")
print(f"  {'Recall@50':<22} {mean('recall_50'):>8.4f}  (ceiling  85%)")
print(f"  {'Recall@100':<22} {mean('recall_100'):>8.4f}  (ceiling  94%)")
print(f"  {'Recall@500':<22} {mean('recall_500'):>8.4f}  (ceiling 100%)")
print(f"  {'-'*32}")
print(f"  {'Success@1':<22} {mean('success_1'):>8.4f}")
print(f"  {'Success@5':<22} {mean('success_5'):>8.4f}")
print(f"  {'Success@10':<22} {mean('success_10'):>8.4f}")
print(f"  {'-'*32}")
print(f"  Best-relevant-doc rank:")
for thresh in [1, 5, 10, 50, 100, 500]:
    n = int((best_ranks_arr <= thresh).sum())
    print(f"    Top-{thresh:<5} {n:>3}/{len(query_ids)}  ({n/len(query_ids):.0%})")
print(f"{'='*62}")

# ── Per-query breakdown ───────────────────────────────────────────────────────

print(f"\n  {'qid':<8} {'n_rel':>5}  {'NDCG@10':>8}  {'R@10':>7}  {'R@50':>7}  {'R@100':>7}  {'R@500':>7}  {'best_rank':>9}")
print(f"  {'-'*70}")
for qid in query_ids:
    v        = results.get(qid, {})
    rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
    ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
    rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
    ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
    best     = min(ranks) if ranks else -1
    print(f"  {qid:<8} {len(qrels[qid]):>5}  "
          f"{v.get('ndcg_cut_10',0):>8.4f}  "
          f"{v.get('recall_10',0):>7.4f}  "
          f"{v.get('recall_50',0):>7.4f}  "
          f"{v.get('recall_100',0):>7.4f}  "
          f"{v.get('recall_500',0):>7.4f}  "
          f"{best:>9}")

# ── Save per-query results ────────────────────────────────────────────────────

out_name = f"results_{args.provider}_{model_slug}_{dataset_dir.name}.jsonl"
out_path = dataset_dir / out_name
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
        "recall@10":  round(v.get("recall_10",    0), 4),
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
