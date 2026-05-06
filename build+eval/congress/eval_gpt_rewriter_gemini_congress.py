"""
eval_gpt_rewriter_gemini_congress.py
=====================================
GPT-5.2 query rewriter + Gemini dense retrieval for the Congressional Hearing ToT benchmark.

Pipeline:
  1. GPT-5.2 reads each fuzzy ToT query and rewrites it into document-language
     that a dense embedding model can match against hearing transcripts
  2. Gemini embeds the rewritten query
  3. Retrieve top-K from pre-embedded corpus

This tests whether an LLM can bridge the gap between memory-language and
transcript-language — if rewrites succeed, the bottleneck is translation,
not retrieval.

Usage:
    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...
    python eval_gpt_rewriter_gemini_congress.py --benchmark-dir congress_corpus_data/beir_export/
    python eval_gpt_rewriter_gemini_congress.py --benchmark-dir congress_corpus_data/beir_export/ --limit 5
"""

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
from tqdm import tqdm
from openai import AsyncOpenAI

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark-dir", required=True)
parser.add_argument("--tag",           default="tot")
parser.add_argument("--model",         default="gpt-5.2")
parser.add_argument("--gemini-model",  default="gemini-embedding-2-preview")
parser.add_argument("--top-k",         type=int, default=1000)
parser.add_argument("--k-values",      default="10,50,100,1000")
parser.add_argument("--ckpt",          default="checkpoints/gpt_rewriter_gemini_congress_cache.jsonl")
parser.add_argument("--sleep",         type=float, default=0.5)
parser.add_argument("--concurrency",   type=int, default=20)
parser.add_argument("--embed-batch",   type=int, default=50)
parser.add_argument("--dim",           type=int, default=3072)
parser.add_argument("--corpus-cache",  default=None,
                    help="Dir with cached Gemini corpus embeddings (default: <benchmark-dir>/gemini_cache)")
parser.add_argument("--chunk-size",    type=int, default=500)
parser.add_argument("--limit",         type=int, default=None)
args = parser.parse_args()

k_values  = [int(k) for k in args.k_values.split(",")]
bench_dir = Path(args.benchmark_dir)
ckpt_path = Path(args.ckpt)
ckpt_path.parent.mkdir(parents=True, exist_ok=True)

corpus_cache_dir = Path(args.corpus_cache) if args.corpus_cache else bench_dir / "gemini_cache"
model_slug = args.gemini_model.replace("/", "_").replace("-", "_")

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
You are an advanced retrieval system. You will be given a tip of tongue query 
describing a users hazy memory of a specific moment from a US congressional hearing. 
They wrote a vague description of what they remember.

Rewrite their description as a search query that is more likely to match the 
actual hearing transcript. Use the kind of language that would appear in an 
official transcript based on whatever you can infer from their description.

Return ONLY a JSON object:
{
  "rewritten_query": "<your rewritten query>",
  "reasoning": "<what you think they're describing and why>"
}"""

REWRITE_USER = """\
{query}

Rewrite this to better match the actual hearing transcript."""

# ── Load benchmark ────────────────────────────────────────────────────────────

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

print("[1/5] Loading benchmark...")
corpus  = load_jsonl(bench_dir / "corpus.jsonl")
queries = load_jsonl(bench_dir / "queries.jsonl")

# Qrels
qrels = {}
with open(bench_dir / "qrels.tsv") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('query-id') or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) == 3:
            qid, cid, score = parts
        elif len(parts) == 4:
            qid, _, cid, score = parts
        else:
            continue
        qrels.setdefault(qid, {})[cid] = int(score)
print(f"  Qrels: {sum(len(v) for v in qrels.values())} pairs across {len(qrels)} queries")

# Query metadata
query_witness = {}
query_memorability = {}
for qid, q in queries.items():
    meta = q.get('metadata', {})
    query_witness[qid] = meta.get('source_speaker', 'unknown').lower()
    query_memorability[qid] = meta.get('memorability', 0)

corpus_ids   = list(corpus.keys())
corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
cid_to_idx   = {cid: i for i, cid in enumerate(corpus_ids)}

query_ids = list(queries.keys())
if args.limit:
    query_ids = query_ids[:args.limit]

print(f"  Corpus:  {len(corpus_ids)} passages")
print(f"  Queries: {len(query_ids)}")

# ── Load / build Gemini corpus embeddings ─────────────────────────────────────

print(f"\n[2/5] Loading Gemini corpus embeddings from {corpus_cache_dir}...")

def load_corpus_embeddings():
    """Load cached corpus embeddings from gemini_cache (from eval_gemini_congress.py)."""
    chunks = list(range(0, len(corpus_ids), args.chunk_size))
    existing = {}
    missing = []

    for cs in chunks:
        p = corpus_cache_dir / f"corpus_{model_slug}_chunk{cs}.npy"
        if p.exists():
            existing[cs] = p
        else:
            missing.append(cs)

    print(f"  Chunks: {len(chunks)} total, {len(existing)} cached, {len(missing)} missing")

    if missing:
        print(f"  Embedding {len(missing)} missing corpus chunks...")
        for cs in tqdm(missing, desc="Corpus embed"):
            chunk_texts = corpus_texts[cs: cs + args.chunk_size]
            chunk_embs = []
            for i in range(0, len(chunk_texts), args.embed_batch):
                batch = chunk_texts[i: i + args.embed_batch]
                chunk_embs.append(embed_texts_gemini(batch, "RETRIEVAL_DOCUMENT"))
                time.sleep(args.sleep)
            chunk_arr = np.concatenate(chunk_embs, axis=0)
            norms = np.linalg.norm(chunk_arr, axis=1, keepdims=True)
            chunk_arr = chunk_arr / np.clip(norms, 1e-12, None)
            p = corpus_cache_dir / f"corpus_{model_slug}_chunk{cs}.npy"
            np.save(p, chunk_arr)
            existing[cs] = p

    out = [np.load(existing[cs]) for cs in chunks]
    return np.concatenate(out, axis=0)


def embed_texts_gemini(texts, task_type="RETRIEVAL_QUERY"):
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


corpus_embeddings = load_corpus_embeddings()

if corpus_embeddings.shape[0] != len(corpus_ids):
    raise SystemExit(
        f"Corpus embeddings {corpus_embeddings.shape[0]} != corpus {len(corpus_ids)}. "
        f"Fix: python fix_gemini_chunks_v2.py --benchmark_dir {bench_dir}"
    )
print(f"  Loaded: {corpus_embeddings.shape}")

# ── GPT rewrite helper ───────────────────────────────────────────────────────

async def call_gpt_async(system_prompt, user_prompt):
    for attempt in range(5):
        try:
            resp = await openai_client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
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

# ── Retrieval helper ──────────────────────────────────────────────────────────

def retrieve_top_k(query_embedding, top_k):
    scores = (query_embedding @ corpus_embeddings.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(corpus_ids[i], float(scores[i])) for i in top_idx]

# ── Load cache ────────────────────────────────────────────────────────────────

print(f"\n[3/5] Loading checkpoint...")
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

print(f"\n[4/5] Running rewrite + retrieve pipeline...")

run = {}
ranked_lists = {}
results_list = []

pending_qids = []
for qid in query_ids:
    if qid in cached:
        rec = cached[qid]
        results_list.append(rec)
        run[qid] = {item[0]: item[1] for item in rec.get("top_results", [])}
        ranked_lists[qid] = [item[0] for item in rec.get("top_results", [])]
    else:
        pending_qids.append(qid)

print(f"  Cached: {len(cached)}")
print(f"  Pending: {len(pending_qids)}")


async def rewrite_one(qid, sem):
    async with sem:
        query_text = queries[qid]["text"]
        rewrite_prompt = REWRITE_USER.format(query=query_text)
        result = await call_gpt_async(REWRITE_SYSTEM, rewrite_prompt)
        if result:
            rewritten = result.get("rewritten_query", query_text)
            reasoning = result.get("reasoning", "")
        else:
            rewritten = query_text
            reasoning = "Rewrite failed, using original"
        return qid, query_text, rewritten, reasoning


async def run_rewrites(pending_qids, concurrency):
    sem = asyncio.Semaphore(concurrency)
    tasks = [rewrite_one(qid, sem) for qid in pending_qids]
    results = []
    pbar = tqdm(total=len(tasks), desc="GPT rewrite")
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
        pbar.update(1)
    pbar.close()
    by_qid = {r[0]: r for r in results}
    return [by_qid[q] for q in pending_qids]


if pending_qids:
    # Phase 1: concurrent GPT rewrites
    print(f"\n  Phase 1: concurrent GPT rewrites ({args.concurrency} at a time)...")
    rewrite_results = asyncio.run(run_rewrites(pending_qids, args.concurrency))

    # Phase 2: batched Gemini embedding of rewritten queries
    print(f"\n  Phase 2: batched Gemini embedding ({args.embed_batch}/batch)...")
    rewritten_queries = [r[2] for r in rewrite_results]
    all_query_embs = []
    for i in tqdm(range(0, len(rewritten_queries), args.embed_batch), desc="Gemini embed"):
        batch = rewritten_queries[i: i + args.embed_batch]
        emb = embed_texts_gemini(batch, task_type="RETRIEVAL_QUERY")
        all_query_embs.append(emb)
        time.sleep(args.sleep)
    all_query_embs_arr = np.concatenate(all_query_embs, axis=0)
    # Normalize
    norms = np.linalg.norm(all_query_embs_arr, axis=1, keepdims=True)
    all_query_embs_arr = all_query_embs_arr / np.clip(norms, 1e-12, None)

    # Phase 3: score + cache
    print(f"\n  Phase 3: scoring + caching...")
    cache_file = open(ckpt_path, "a")
    for i, (qid, orig_text, rewritten, reasoning) in enumerate(tqdm(rewrite_results, desc="Scoring")):
        retrieved = retrieve_top_k(all_query_embs_arr[i], args.top_k)
        run[qid] = {did: score for did, score in retrieved}
        ranked_lists[qid] = [did for did, _ in retrieved]

        rec = {
            "query_id":        qid,
            "original_query":  orig_text,
            "rewritten_query": rewritten,
            "reasoning":       reasoning,
            "ranked":     retrieved,
        }
        results_list.append(rec)
        cache_file.write(json.dumps(rec) + "\n")
        cache_file.flush()
    cache_file.close()

# ── Evaluate ──────────────────────────────────────────────────────────────────

print(f"\n[5/5] Evaluating...")

metrics = set()
for k in k_values:
    metrics.add(f"ndcg_cut_{k}")
    metrics.add(f"recall_{k}")
metrics.add("map")
metrics.add("recip_rank")

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
per_query = evaluator.evaluate(run)

ordered = (
    [f"ndcg_cut_{k}" for k in k_values] +
    [f"recall_{k}"   for k in k_values] +
    ["map", "recip_rank"]
)

agg = defaultdict(list)
for qid, ms in per_query.items():
    for m, v in ms.items():
        agg[m].append(v)

print(f"\n{'='*60}")
print(f"  {args.model} Rewriter + Gemini — Congress ToT")
print(f"  Top-K: {args.top_k}  |  Queries: {len(per_query)}")
print(f"{'='*60}")
print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8}")
print(f"  {'-'*52}")
for m in ordered:
    if m in agg:
        vals = agg[m]
        print(f"  {m:<25} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

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

print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""
gpt_slug = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
out_path = bench_dir / f"gpt_rewriter_gemini_{gpt_slug}{tag_suffix}_results.jsonl"

rows_out = []
for qid in query_ids:
    v = per_query.get(qid, {})
    rec = next((r for r in results_list if r["query_id"] == qid), {})
    rows_out.append({
        "query_id":        qid,
        "witness":         query_witness.get(qid, "unknown"),
        "memorability":    query_memorability.get(qid, 0),
        "original_query":  queries[qid]["text"],
        "rewritten_query": rec.get("rewritten_query", ""),
        "reasoning":       rec.get("reasoning", ""),
        "ndcg@10":         round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50":         round(v.get("ndcg_cut_50", 0), 4),
        "recall@10":       round(v.get("recall_10",   0), 4),
        "recall@50":       round(v.get("recall_50",   0), 4),
        "recall@100":      round(v.get("recall_100",  0), 4),
        "recall@1000":     round(v.get("recall_1000", 0), 4),
        "map":             round(v.get("map",          0), 4),
        "mrr":             round(v.get("recip_rank",   0), 4),
        "ranked_list":     ranked_lists.get(qid, []),
    })
out_path.write_text("\n".join(json.dumps(r) for r in rows_out))

summary = {
    "model":       args.model,
    "retriever":   args.gemini_model,
    "algorithm":   "gpt_rewriter",
    "tag":         args.tag,
    "corpus_size": len(corpus_ids),
    "n_queries":   len(per_query),
    "metrics":     {m: round(float(np.mean(agg.get(m, [0]))), 4) for m in ordered},
    "per_memorability_ndcg10": {
        str(m): round(float(np.mean(v)), 4) for m, v in sorted(mem_ndcg10.items(), reverse=True)
    },
}
summary_path = bench_dir / f"gpt_rewriter_gemini_{gpt_slug}{tag_suffix}_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[+] Results → {out_path}")
print(f"[+] Summary → {summary_path}")
print(f"[+] Cache   → {ckpt_path}")
