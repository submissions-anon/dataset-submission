"""
eval_gpt_rewriter_gemini.py
===========================
GPT-5.2 query rewriter + Gemini dense retrieval for Twitter benchmark.

Simple pipeline:
  1. GPT-5.2 rewrites the query once
  2. Gemini retrieves top-1000 with the rewritten query

Usage:
  export OPENAI_API_KEY=...
  export GEMINI_API_KEY=...
  python eval_gpt_rewriter_gemini.py
  python eval_gpt_rewriter_gemini.py --limit 5  # smoke test
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm
from openai import OpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset")
parser.add_argument("--model", default="gpt-5.2", help="OpenAI model for rewriting")
parser.add_argument("--gemini-model", default="gemini-embedding-2-preview")
parser.add_argument("--top-k", type=int, default=1000, help="Retrieval depth")
parser.add_argument("--ckpt", default="checkpoints/gpt_rewriter_gemini_cache.jsonl")
parser.add_argument("--sleep", type=float, default=0.5)
parser.add_argument("--limit", type=int, default=None, help="Only run N queries (smoke test)")
parser.add_argument("--dim", type=int, default=3072, help="Gemini embedding dimension")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
corpus_file = dataset_dir / "corpus_full.jsonl"
queries_file = dataset_dir / "queries.jsonl"
qrels_file = dataset_dir / "qrels.tsv"

for f in [corpus_file, queries_file, qrels_file]:
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

# ── Clients ───────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")
openai_client = OpenAI(api_key=api_key)

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise SystemExit("Set GEMINI_API_KEY env var")

try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    raise ImportError("pip install google-genai")
gemini_client = genai.Client(api_key=gemini_api_key)

# ── Prompts ───────────────────────────────────────────────────────────────────

REWRITE_SYSTEM = """\
You are an expert at reformulating queries to improve retrieval of relevant tweets.

Given a query describing an implicit theme or pattern in tweets, rewrite it to better
match how such tweets would actually be written. Focus on the language, phrasing, and
keywords that would appear in relevant tweets.

Return ONLY a JSON object:
{
  "rewritten_query": "<your rewritten query>",
  "rationale": "<brief explanation>"
}"""

REWRITE_USER = """\
ORIGINAL QUERY:
{query}

Rewrite this query to improve tweet retrieval."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_corpus(path):
    corpus = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                corpus[doc["_id"]] = doc.get("text", "")
    return corpus


def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                queries[q["_id"]] = q.get("text", "")
    return queries


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], parts[2]
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def call_gpt(system_prompt, user_prompt):
    """Call GPT with retry logic."""
    for attempt in range(5):
        try:
            resp = openai_client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                max_completion_tokens=128000,
            )
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                raise ValueError("Empty response")

            clean = raw.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = "\n".join(clean.split("\n")[:-1])
            return json.loads(clean.strip())

        except KeyboardInterrupt:
            raise
        except Exception as e:
            wait = 10 * (2 ** attempt)
            print(f"\n  [GPT attempt {attempt+1}/5, retry in {wait}s] {e}")
            time.sleep(wait)
    return None


# ── Gemini Embedding ──────────────────────────────────────────────────────────

def embed_texts_gemini(texts, task_type="RETRIEVAL_QUERY"):
    """Embed texts with Gemini."""
    for attempt in range(6):
        try:
            cfg = gtypes.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=args.dim,
            )
            result = gemini_client.models.embed_content(
                model=args.gemini_model,
                contents=texts,
                config=cfg,
            )
            time.sleep(0.3)
            return np.array([e.values for e in result.embeddings], dtype=np.float32)
        except Exception as e:
            wait = 30 * (2 ** attempt)
            print(f"\n  [Gemini attempt {attempt+1}/6, retry in {wait}s] {e}")
            time.sleep(wait)
    raise RuntimeError("Gemini embedding failed after 6 attempts")


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def retrieve_top_k(query_embedding, corpus_embeddings, corpus_ids, top_k):
    """Retrieve top-K documents by cosine similarity."""
    scores = query_embedding @ corpus_embeddings.T
    scores = scores.flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus_ids[i], float(scores[i])) for i in top_indices]


# ── Load Data ─────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  GPT-5.2 Query Rewriter + Gemini Retrieval (Twitter)")
print(f"  Model: {args.model}  |  Gemini: {args.gemini_model}")
print(f"  Top-K: {args.top_k}")
print(f"{'='*70}")

print("\n[1/5] Loading corpus...")
corpus = load_corpus(corpus_file)
corpus_ids = list(corpus.keys())
print(f"  {len(corpus_ids):,} tweets")

print("\n[2/5] Loading queries & qrels...")
queries = load_queries(queries_file)
qrels = load_qrels(qrels_file)
query_ids = [qid for qid in queries if qid in qrels]

if args.limit:
    query_ids = query_ids[:args.limit]
    print(f"  Smoke test: limiting to {args.limit} queries")

print(f"  {len(query_ids)} queries with qrels")

print("\n[3/5] Loading corpus embeddings...")
ckpt_dir = Path("eval_cache_gemini")
model_slug = args.gemini_model.replace("/", "_").replace("-", "_")
corpus_emb_path = ckpt_dir / f"corpus_{model_slug}_full_tweet.npy"

if corpus_emb_path.exists():
    print(f"  Loading from {corpus_emb_path}")
    corpus_embeddings = normalize(np.load(corpus_emb_path))
else:
    raise FileNotFoundError(f"Corpus embeddings not found at {corpus_emb_path}")
print(f"  Embeddings shape: {corpus_embeddings.shape}")

# ── Load checkpoint ───────────────────────────────────────────────────────────

print("\n[4/5] Loading checkpoint...")
ckpt_path = Path(args.ckpt)
ckpt_path.parent.mkdir(exist_ok=True)
cached = {}
if ckpt_path.exists():
    with open(ckpt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cached[rec["query_id"]] = rec
    print(f"  Loaded {len(cached)} cached results")
else:
    print(f"  No checkpoint found, starting fresh")

# ── Run pipeline ──────────────────────────────────────────────────────────────

print(f"\n[5/5] Running rewrite + retrieve pipeline...")

run = {}
results_list = []
cache_file = open(ckpt_path, "a")

for qid in tqdm(query_ids, desc="Processing"):
    if qid in cached:
        rec = cached[qid]
        results_list.append(rec)
        # Rebuild run from cached
        run[qid] = {item[0]: item[1] for item in rec.get("top_results", [])}
        continue

    query_text = queries[qid]

    # Step 1: Rewrite query
    rewrite_prompt = REWRITE_USER.format(query=query_text)
    rewrite_result = call_gpt(REWRITE_SYSTEM, rewrite_prompt)

    if rewrite_result:
        rewritten_query = rewrite_result.get("rewritten_query", query_text)
        rationale = rewrite_result.get("rationale", "")
    else:
        rewritten_query = query_text
        rationale = "Rewrite failed, using original"

    # Step 2: Embed rewritten query and retrieve
    query_emb = normalize(embed_texts_gemini([rewritten_query], "RETRIEVAL_QUERY"))
    retrieved = retrieve_top_k(query_emb[0], corpus_embeddings, corpus_ids, args.top_k)

    # Build run dict
    run[qid] = {did: score for did, score in retrieved}

    # Save to cache
    rec = {
        "query_id": qid,
        "original_query": query_text,
        "rewritten_query": rewritten_query,
        "rationale": rationale,
        "top_results": retrieved[:100],  # Save top 100 for inspection
    }
    results_list.append(rec)
    cache_file.write(json.dumps(rec) + "\n")
    cache_file.flush()

    time.sleep(args.sleep)

cache_file.close()

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
    vals = [v.get(key, 0.0) for v in results.values()]
    return float(np.mean(vals)) if vals else 0.0


# ── Print results ─────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  {args.model} Rewriter + Gemini  (Twitter)")
print(f"  Top-K: {args.top_k}  |  Queries: {len(query_ids)}")
print(f"{'='*70}")
print(f"  {'Metric':<22} {'Score':>8}")
print(f"  {'-'*32}")
print(f"  {'MRR':<22} {mean('recip_rank'):>8.4f}")
print(f"  {'NDCG@10':<22} {mean('ndcg_cut_10'):>8.4f}  <- primary")
print(f"  {'NDCG@50':<22} {mean('ndcg_cut_50'):>8.4f}")
print(f"  {'NDCG@100':<22} {mean('ndcg_cut_100'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Recall@10':<22} {mean('recall_10'):>8.4f}")
print(f"  {'Recall@50':<22} {mean('recall_50'):>8.4f}")
print(f"  {'Recall@100':<22} {mean('recall_100'):>8.4f}")
print(f"  {'Recall@500':<22} {mean('recall_500'):>8.4f}")
print(f"  {'-'*32}")
print(f"  {'Success@1':<22} {mean('success_1'):>8.4f}")
print(f"  {'Success@5':<22} {mean('success_5'):>8.4f}")
print(f"  {'Success@10':<22} {mean('success_10'):>8.4f}")
print(f"{'='*70}")

# ── Per-query breakdown ───────────────────────────────────────────────────────

print(f"\n  {'qid':<10} {'n_rel':>5}  {'NDCG@10':>8}  {'R@10':>7}  {'R@50':>7}  {'R@100':>7}  {'R@500':>7}")
print(f"  {'-'*65}")
for qid in query_ids[:50]:  # Show first 50
    v = results.get(qid, {})
    print(f"  {qid:<10} {len(qrels[qid]):>5}  "
          f"{v.get('ndcg_cut_10', 0):>8.4f}  "
          f"{v.get('recall_10', 0):>7.4f}  "
          f"{v.get('recall_50', 0):>7.4f}  "
          f"{v.get('recall_100', 0):>7.4f}  "
          f"{v.get('recall_500', 0):>7.4f}")

# ── Save results ──────────────────────────────────────────────────────────────

model_slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
out_path = dataset_dir / f"results_gpt_rewriter_gemini_{model_slug}.jsonl"

rows = []
for qid in query_ids:
    v = results.get(qid, {})
    rec = next((r for r in results_list if r["query_id"] == qid), {})
    rows.append({
        "query_id": qid,
        "original_query": queries[qid][:150],
        "rewritten_query": rec.get("rewritten_query", "")[:150],
        "mrr": round(v.get("recip_rank", 0), 4),
        "ndcg@10": round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50": round(v.get("ndcg_cut_50", 0), 4),
        "recall@10": round(v.get("recall_10", 0), 4),
        "recall@50": round(v.get("recall_50", 0), 4),
        "recall@100": round(v.get("recall_100", 0), 4),
        "recall@500": round(v.get("recall_500", 0), 4),
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows))
print(f"\n[+] Results -> {out_path}")
print(f"[+] Cache   -> {ckpt_path}")

# ── Final Summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  SUMMARY: {args.model} Rewriter + Gemini (Twitter)")
print(f"{'='*70}")
print(f"  NDCG@10:    {mean('ndcg_cut_10'):.4f}")
print(f"  NDCG@50:    {mean('ndcg_cut_50'):.4f}")
print(f"  Recall@10:  {mean('recall_10'):.4f}")
print(f"  Recall@50:  {mean('recall_50'):.4f}")
print(f"  Recall@100: {mean('recall_100'):.4f}")
print(f"  Recall@500: {mean('recall_500'):.4f}")
print(f"  MRR:        {mean('recip_rank'):.4f}")
print(f"{'='*70}")
