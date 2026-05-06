"""
eval_gpt_rewriter_gemini.py
===========================
GPT-5.2 query rewriter + Gemini dense retrieval for Writing Analogues (author attribution) benchmark.

Simple pipeline:
  1. GPT-5.2 rewrites the query once
  2. Gemini retrieves top-1000 with the rewritten query

Usage:
  export OPENAI_API_KEY=...
  export GEMINI_API_KEY=...
  python eval_gpt_rewriter_gemini.py --corpus-dir corpus
  python eval_gpt_rewriter_gemini.py --corpus-dir corpus --limit 5  # smoke test
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from openai import AsyncOpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark-dir", required=True,
                    help="Directory with corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json")
parser.add_argument("--corpus-dir", required=True,
                    help="Directory containing 1.txt, 2.txt, ... text files")
parser.add_argument("--gold-track", default="corpus_track.csv",
                    help="Path to v2 gold corpus_track.csv (snippet_id, author_name, post_title, post_url). "
                         "Used for per-author breakdown.")
parser.add_argument("--tag", default="v2",
                    help="Suffix for output filenames.")
parser.add_argument("--model", default="gpt-5.2", help="OpenAI model for rewriting")
parser.add_argument("--gemini-model", default="gemini-embedding-2-preview")
parser.add_argument("--top-k", type=int, default=1000, help="Retrieval depth")
parser.add_argument("--ckpt", default="checkpoints/gpt_rewriter_gemini_v2_cache.jsonl")
parser.add_argument("--sleep", type=float, default=0.5,
                    help="Sleep between Gemini batches (rate-limit buffer)")
parser.add_argument("--concurrency", type=int, default=20,
                    help="Max concurrent GPT rewrite calls")
parser.add_argument("--embed-batch", type=int, default=50,
                    help="Rewritten queries per Gemini embedding batch")
parser.add_argument("--limit", type=int, default=None, help="Only run N queries (smoke test)")
parser.add_argument("--dim", type=int, default=3072, help="Gemini embedding dimension")
args = parser.parse_args()

bench_dir = Path(args.benchmark_dir)
corpus_dir = Path(args.corpus_dir)
corpus_jsonl = bench_dir / "corpus.jsonl"
queries_jsonl = bench_dir / "queries.jsonl"
qrels_file = bench_dir / "qrels.tsv"
excluded_ids_file = bench_dir / "per_query_excluded_ids.json"

for f in [corpus_jsonl, queries_jsonl, qrels_file]:
    if not f.exists():
        raise FileNotFoundError(f"Missing: {f}")

# ── Clients ───────────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Set OPENAI_API_KEY env var")
openai_client = AsyncOpenAI(api_key=api_key)

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
You are an expert at author attribution - identifying texts written by the same author.

Given a text snippet, rewrite the query to better capture the distinctive stylistic features
that would help find other texts by the same author. Focus on vocabulary patterns, sentence
structure, tone, and other authorial fingerprints.

Return ONLY a JSON object:
{
  "rewritten_query": "<your rewritten query>",
  "rationale": "<brief explanation>"
}"""

REWRITE_USER = """\
ORIGINAL TEXT:
{query}

Rewrite this to capture the author's distinctive style for finding other texts by the same author."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl_with_text(jsonl_path, text_dir):
    """Load JSONL and resolve text from .txt files."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                sid = d["_id"]
                txt_path = text_dir / f"{sid}.txt"
                if txt_path.exists():
                    d["text"] = txt_path.read_text(encoding="utf-8", errors="replace").strip()
                else:
                    d["text"] = ""
                out[sid] = d
    return out


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


async def call_gpt_async(system_prompt, user_prompt):
    """Async GPT call with retry logic."""
    for attempt in range(5):
        try:
            resp = await openai_client.chat.completions.create(
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
            await asyncio.sleep(wait)
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


def retrieve_top_k(query_embedding, corpus_embeddings, corpus_ids, top_k, exclude_ids=None):
    """Retrieve top-K documents by cosine similarity."""
    scores = query_embedding @ corpus_embeddings.T
    scores = scores.flatten()

    # Mask excluded IDs
    if exclude_ids:
        exclude_set = set(exclude_ids)
        for i, did in enumerate(corpus_ids):
            if did in exclude_set:
                scores[i] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus_ids[i], float(scores[i])) for i in top_indices if scores[i] > -np.inf]


# ── Load Data ─────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  GPT-5.2 Query Rewriter + Gemini Retrieval (Writing Analogues)")
print(f"  Model: {args.model}  |  Gemini: {args.gemini_model}")
print(f"  Top-K: {args.top_k}")
print(f"{'='*70}")

print("\n[1/5] Loading corpus...")
corpus_by_id = load_jsonl_with_text(corpus_jsonl, corpus_dir)
corpus_ids = list(corpus_by_id.keys())
print(f"  {len(corpus_ids):,} snippets")

print("\n[2/5] Loading queries & qrels...")
queries_by_id = load_jsonl_with_text(queries_jsonl, corpus_dir)
qrels = load_qrels(qrels_file)
query_ids = [qid for qid in queries_by_id if qid in qrels]

# Load per-query excluded IDs (same-paper snippets — trivially easy lexically)
per_query_excluded = {}
if excluded_ids_file.exists():
    with open(excluded_ids_file) as f:
        per_query_excluded = json.load(f)
    per_query_excluded = {qid: [str(x) for x in ids] for qid, ids in per_query_excluded.items()}
    print(f"  Loaded per-query exclusions for {len(per_query_excluded)} queries")

# Author metadata for breakdowns (v2 gold CSV)
import csv as _csv
from collections import defaultdict as _defaultdict
snippet_author = {}

csv_path = Path(args.gold_track)
if csv_path.exists():
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        reader.fieldnames = [fn.strip() for fn in reader.fieldnames or []]
        for row in reader:
            sid = (row.get("snippet_id") or "").strip()
            author = (row.get("author_name") or "").strip()
            if sid and author:
                snippet_author[sid] = author
    print(f"  Loaded author metadata for {len(snippet_author)} gold snippets")

if args.limit:
    query_ids = query_ids[:args.limit]
    print(f"  Smoke test: limiting to {args.limit} queries")

print(f"  {len(query_ids)} queries with qrels")

print("\n[3/5] Loading corpus embeddings...")
ckpt_dir = bench_dir / "gemini_cache"
model_slug = args.gemini_model.replace("/", "_").replace("-", "_")

# Load chunked embeddings
chunk_files = sorted(ckpt_dir.glob(f"corpus_{model_slug}_chunk*.npy"))
if chunk_files:
    print(f"  Loading {len(chunk_files)} embedding chunks...")
    chunks = []
    for cf in chunk_files:
        chunks.append(np.load(cf))
    corpus_embeddings = normalize(np.concatenate(chunks, axis=0))
    print(f"  Embeddings shape: {corpus_embeddings.shape}")
else:
    raise FileNotFoundError(
        f"Corpus embeddings not found in {ckpt_dir}. "
        f"Run eval_gemini.py first to generate them."
    )

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

# Separate cached queries from ones we still need to process
pending_qids = []
for qid in query_ids:
    if qid in cached:
        rec = cached[qid]
        results_list.append(rec)
        run[qid] = {item[0]: item[1] for item in rec.get("top_results", [])}
    else:
        pending_qids.append(qid)

print(f"  Cached: {len(cached)}")
print(f"  Pending: {len(pending_qids)}")


async def rewrite_one(qid, sem):
    """Async wrapper: GPT rewrite for a single query, semaphore-bounded."""
    async with sem:
        query_text = queries_by_id[qid]["text"]
        rewrite_prompt = REWRITE_USER.format(query=query_text)
        result = await call_gpt_async(REWRITE_SYSTEM, rewrite_prompt)
        if result:
            rewritten = result.get("rewritten_query", query_text)
            rationale = result.get("rationale", "")
        else:
            rewritten = query_text
            rationale = "Rewrite failed, using original"
        return qid, query_text, rewritten, rationale


async def run_rewrites(pending_qids, concurrency):
    """Run all rewrites concurrently with a semaphore cap."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [rewrite_one(qid, sem) for qid in pending_qids]
    # Use tqdm.asyncio for a live progress bar
    results = []
    pbar = tqdm(total=len(tasks), desc="GPT rewrite")
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        pbar.update(1)
    pbar.close()
    # Reorder to match pending_qids
    by_qid = {r[0]: r for r in results}
    return [by_qid[q] for q in pending_qids]


if pending_qids:
    # Phase 1: concurrent GPT rewrites
    print(f"\n  Phase 1: concurrent GPT rewrites ({args.concurrency} at a time)...")
    rewrite_results = asyncio.run(run_rewrites(pending_qids, args.concurrency))

    # Phase 2: batched Gemini embedding
    print(f"\n  Phase 2: batched Gemini embedding ({args.embed_batch}/batch)...")
    rewritten_queries = [r[2] for r in rewrite_results]
    all_query_embs = []
    for i in tqdm(range(0, len(rewritten_queries), args.embed_batch), desc="Gemini embed"):
        batch = rewritten_queries[i: i + args.embed_batch]
        emb = embed_texts_gemini(batch, task_type="RETRIEVAL_QUERY")
        all_query_embs.append(emb)
        time.sleep(args.sleep)
    all_query_embs = normalize(np.concatenate(all_query_embs, axis=0))

    # Phase 3: score + cache
    print(f"\n  Phase 3: scoring + caching...")
    cache_file = open(ckpt_path, "a")
    for i, (qid, orig_text, rewritten, rationale) in enumerate(tqdm(rewrite_results, desc="Scoring")):
        excluded_ids = per_query_excluded.get(qid, [])
        retrieved = retrieve_top_k(
            all_query_embs[i], corpus_embeddings, corpus_ids, args.top_k,
            exclude_ids=excluded_ids,
        )
        run[qid] = {did: score for did, score in retrieved}
        rec = {
            "query_id":        qid,
            "original_query":  orig_text,
            "rewritten_query": rewritten,
            "rationale":       rationale,
            "top_results":     retrieved[:100],
        }
        results_list.append(rec)
        cache_file.write(json.dumps(rec) + "\n")
        cache_file.flush()
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
print(f"  {args.model} Rewriter + Gemini  (Writing Analogues)")
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

# ── Per-author breakdown ──────────────────────────────────────────────────────

author_ndcg10 = _defaultdict(list)
author_ndcg50 = _defaultdict(list)

for qid in query_ids:
    md = results.get(qid, {})
    a  = snippet_author.get(qid, "unknown")
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

tag_suffix = f"_{args.tag}" if args.tag else ""
model_slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
out_path = bench_dir / f"results_gpt_rewriter_gemini_{model_slug}{tag_suffix}.jsonl"

rows = []
for qid in query_ids:
    v = results.get(qid, {})
    rec = next((r for r in results_list if r["query_id"] == qid), {})
    rows.append({
        "query_id": qid,
        "original_query": queries_by_id[qid]["text"],
        "rewritten_query": rec.get("rewritten_query", ""),
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
print(f"  SUMMARY: {args.model} Rewriter + Gemini (Writing Analogues)")
print(f"{'='*70}")
print(f"  NDCG@10:    {mean('ndcg_cut_10'):.4f}")
print(f"  NDCG@50:    {mean('ndcg_cut_50'):.4f}")
print(f"  Recall@10:  {mean('recall_10'):.4f}")
print(f"  Recall@50:  {mean('recall_50'):.4f}")
print(f"  Recall@100: {mean('recall_100'):.4f}")
print(f"  Recall@500: {mean('recall_500'):.4f}")
print(f"  MRR:        {mean('recip_rank'):.4f}")
print(f"{'='*70}")
