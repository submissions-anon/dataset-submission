"""
eval_qwen3.py
=============
Qwen3-Embedding dense retrieval eval for the Analogues (author attribution) benchmark.

Corpus:  directory of N.txt snippet files
Qrels:   qrels/test.tsv  (same-author, different-paper snippets)
Exclusion: per_query_excluded_ids.json (same-paper snippets removed at retrieval time)

Checkpoints corpus/query embeddings to disk — restarts resume from last chunk.

Usage:
  # 0.6B
  python eval_qwen3.py \
      --corpus_dir /path/to/corpus/ \
      --model Qwen/Qwen3-Embedding-0.6B

  # 4B (needs ~20GB VRAM)
  python eval_qwen3.py \
      --corpus_dir /path/to/corpus/ \
      --model Qwen/Qwen3-Embedding-4B \
      --batch_size 4
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark_dir", required=True,
                    help="Directory containing corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json")
parser.add_argument("--corpus_dir",    required=True,
                    help="Directory containing 1.txt, 2.txt, ...")
parser.add_argument("--gold_track",    default="corpus_track.csv",
                    help="Path to v2 gold CSV (snippet_id, author_name, post_title, post_url). "
                         "Used for per-author breakdown.")
parser.add_argument("--tag",         default="v2",
                    help="Suffix for output filenames.")
parser.add_argument("--model",       default="Qwen/Qwen3-Embedding-0.6B")
parser.add_argument("--use_instruction", action="store_true",
                    help="Prepend Qwen3's native instruction format to queries. "
                         "Qwen3-Embedding was trained with this format; may help. "
                         "(On Gemini the equivalent prefix HURT retrieval for v2 — test both.)")
parser.add_argument("--batch_size",  type=int, default=16)
parser.add_argument("--max_length",  type=int, default=999999)
parser.add_argument("--top_k",       type=int, default=1000)
parser.add_argument("--chunk_size",  type=int, default=500,
                    help="Checkpoint corpus embeddings every N docs")
parser.add_argument("--ckpt_dir",    default=None,
                    help="Embedding cache dir (default: <benchmark_dir>/qwen3_cache)")
parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

bench_dir = Path(args.benchmark_dir)
ckpt_dir  = Path(args.ckpt_dir) if args.ckpt_dir else bench_dir / "qwen3_cache"
ckpt_dir.mkdir(exist_ok=True, parents=True)
model_slug = args.model.split("/")[-1]
# Include instruction flag in cache prefix so with/without caches don't collide
cache_suffix = "_instr" if args.use_instruction else "_noinstr"

QUERY_INSTRUCTION = (
    "Given a text snippet, retrieve other text snippets written by the same author."
)

# ── Load benchmark files ───────────────────────────────────────────────────────

def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d['_id']] = d
    return out

print("[1/5] Loading benchmark...")

corpus  = load_jsonl(bench_dir / "corpus.jsonl")
queries = load_jsonl(bench_dir / "queries.jsonl")

corpus_dir = Path(args.corpus_dir)
for sid, doc in corpus.items():
    doc['text'] = (corpus_dir / f"{sid}.txt").read_text(encoding="utf-8", errors="replace").strip()
for sid, q in queries.items():
    q['text'] = (corpus_dir / f"{sid}.txt").read_text(encoding="utf-8", errors="replace").strip()

# Binary qrels
qrels = {}
with open(bench_dir / "qrels.tsv") as f:
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
with open(bench_dir / "per_query_excluded_ids.json") as f:
    excluded = json.load(f)
excluded = {qid: set(str(x) for x in ids) for qid, ids in excluded.items()}

# ordered corpus for matrix ops
corpus_ids   = list(corpus.keys())
corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
cid_to_idx   = {cid: i for i, cid in enumerate(corpus_ids)}

query_ids   = list(queries.keys())
query_texts = [queries[qid]['text'] for qid in query_ids]

print(f"  Corpus:  {len(corpus_ids)} snippets")
print(f"  Queries: {len(query_ids)}")

# ── Load per-author metadata (v2 gold CSV) ───────────────────────────────────

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
else:
    print(f"  [!] {csv_path} not found — skipping per-author breakdown")

# ── Encoding helpers ───────────────────────────────────────────────────────────

print(f"\n[2/5] Loading model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model     = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
model     = model.to(args.device).eval()


def last_token_pool(hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[
        torch.arange(hidden_states.shape[0], device=hidden_states.device),
        seq_lengths,
    ]


def encode_batch(texts, instruction=None):
    if instruction:
        texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    ).to(args.device)
    with torch.no_grad():
        out = model(**enc)
    embs = last_token_pool(out.last_hidden_state, enc["attention_mask"])
    return F.normalize(embs, p=2, dim=-1).cpu().float().numpy()


def encode_with_checkpoint(texts, ckpt_prefix, instruction=None):
    total  = len(texts)
    chunks = list(range(0, total, args.chunk_size))
    out_embs = []

    last_done = -1
    for chunk_start in chunks:
        p = ckpt_dir / f"{ckpt_prefix}_{model_slug}_chunk{chunk_start}.npy"
        if p.exists():
            out_embs.append(np.load(p))
            last_done = chunk_start
        else:
            break

    if out_embs and last_done == chunks[-1]:
        print(f"  Loaded all {len(chunks)} chunks from cache")
        return np.concatenate(out_embs, axis=0)

    if out_embs:
        resume_from = chunks[len(out_embs)]
        print(f"  Resuming from chunk {resume_from} ({len(out_embs)} cached)")
    else:
        resume_from = 0

    for chunk_start in tqdm(chunks, desc=f"Encoding {ckpt_prefix}"):
        if chunk_start < resume_from:
            continue
        chunk_texts = texts[chunk_start: chunk_start + args.chunk_size]
        chunk_embs  = []
        for i in range(0, len(chunk_texts), args.batch_size):
            batch = chunk_texts[i: i + args.batch_size]
            chunk_embs.append(encode_batch(batch, instruction=instruction))
        chunk_arr  = np.concatenate(chunk_embs, axis=0)
        chunk_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}_chunk{chunk_start}.npy"
        np.save(chunk_path, chunk_arr)
        out_embs.append(chunk_arr)
        torch.cuda.empty_cache()

    return np.concatenate(out_embs, axis=0)

# ── Encode ────────────────────────────────────────────────────────────────────

print("\n[3/5] Encoding corpus...")
# Corpus never gets the instruction prefix — only queries do.
corp_embs = encode_with_checkpoint(corpus_texts, "corpus")

if corp_embs.shape[0] != len(corpus_ids):
    raise SystemExit(
        f"\n[error] Corpus embedding cache has {corp_embs.shape[0]} rows but "
        f"current corpus has {len(corpus_ids)} snippets.\n"
        f"Fix: rm {ckpt_dir}/corpus_*.npy  and rerun."
    )

print(f"\n[4/5] Encoding queries... (use_instruction={args.use_instruction})")
# Queries key the cache by whether we used the instruction prefix, so you can
# A/B test: `--use_instruction` vs no flag caches to different files.
query_cache_prefix = f"queries{cache_suffix}"
query_embs = encode_with_checkpoint(
    query_texts,
    query_cache_prefix,
    instruction=QUERY_INSTRUCTION if args.use_instruction else None,
)

if query_embs.shape[0] != len(query_ids):
    raise SystemExit(
        f"\n[error] Query embedding cache has {query_embs.shape[0]} rows but "
        f"current benchmark has {len(query_ids)} queries.\n"
        f"Fix: rm {ckpt_dir}/{query_cache_prefix}_*.npy  and rerun."
    )

# ── Retrieve with per-query exclusion ─────────────────────────────────────────

print(f"\n[5/5] Retrieving (top_k={args.top_k})...")

SCORE_BATCH  = 256
corp_embs_t  = torch.from_numpy(corp_embs).to(args.device)
run          = {}  # qid -> {cid: score}

for i in tqdm(range(0, len(query_ids), SCORE_BATCH), desc="Scoring"):
    q_batch  = torch.from_numpy(query_embs[i: i + SCORE_BATCH]).to(args.device)
    scores   = (q_batch @ corp_embs_t.T).cpu().float().numpy()  # (batch, corpus)

    for j, qid in enumerate(query_ids[i: i + SCORE_BATCH]):
        row  = scores[j].copy()
        excl = excluded.get(qid, set())

        # mask excluded docs to -inf before ranking
        for cid in excl:
            idx = cid_to_idx.get(cid)
            if idx is not None:
                row[idx] = -np.inf

        top_idx  = np.argsort(row)[::-1][:args.top_k]
        run[qid] = {corpus_ids[k]: float(row[k]) for k in top_idx if row[k] > -np.inf}

del corp_embs_t
torch.cuda.empty_cache()

# ── Evaluate ──────────────────────────────────────────────────────────────────

k_values = [10, 50, 100, 1000]
metrics  = set()
for k in k_values:
    metrics.add(f"ndcg_cut_{k}")
    metrics.add(f"recall_{k}")
metrics.add("map")

evaluator  = pytrec_eval.RelevanceEvaluator(qrels, metrics)
per_query  = evaluator.evaluate(run)

def mean_metric(key):
    return np.mean([v.get(key, 0.0) for v in per_query.values()])

# ── Print overall results ──────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  Qwen3-Embedding  —  Analogues Benchmark")
print(f"{'='*60}")
print(f"  Model:   {model_slug}")
print(f"  Corpus:  {len(corpus_ids)} snippets")
print(f"  Queries: {len(query_ids)}")
print(f"  {'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8}")
print(f"  {'-'*46}")

ordered = (
    [f"ndcg_cut_{k}" for k in k_values] +
    [f"recall_{k}"   for k in k_values] +
    ["map"]
)
for metric in ordered:
    vals = [v.get(metric, 0.0) for v in per_query.values()]
    print(f"  {metric:<20} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

# ── Per-author breakdown ───────────────────────────────────────────────────────

author_scores    = defaultdict(list)
author_scores_50 = defaultdict(list)

for qid, metrics_d in per_query.items():
    author = snippet_author.get(qid, "unknown")
    author_scores[author].append(metrics_d.get("ndcg_cut_10", 0.0))
    author_scores_50[author].append(metrics_d.get("ndcg_cut_50", 0.0))

if snippet_author:
    print(f"\n  {'─'*56}")
    print(f"  Per-author nDCG@10")
    print(f"  {'─'*56}")
    print(f"  {'Author':<30} {'nDCG@10':>8} {'nDCG@50':>8} {'n':>5}")
    print(f"  {'─'*56}")
    for author in sorted(author_scores, key=lambda a: -np.mean(author_scores[a])):
        vals   = author_scores[author]
        vals50 = author_scores_50[author]
        print(f"  {author[:29]:<30} {np.mean(vals):>8.4f} {np.mean(vals50):>8.4f} {len(vals):>5}")

print(f"{'='*60}")

# ── Save per-query results ─────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""
out_path = bench_dir / f"qwen3_{model_slug}{tag_suffix}{cache_suffix}_results.jsonl"
rows_out = []
for qid in query_ids:
    v = per_query.get(qid, {})
    # Get ranked list sorted by score descending
    qrun = run.get(qid, {})
    ranked_list = sorted(qrun.keys(), key=lambda cid: qrun[cid], reverse=True)
    rows_out.append({
        "query_id":    qid,
        "author":      snippet_author.get(qid, "unknown"),
        "ndcg@10":     round(v.get("ndcg_cut_10", 0), 4),
        "ndcg@50":     round(v.get("ndcg_cut_50", 0), 4),
        "recall@10":   round(v.get("recall_10",   0), 4),
        "recall@50":   round(v.get("recall_50",   0), 4),
        "map":         round(v.get("map",          0), 4),
        "ranked_list": ranked_list,
    })

out_path.write_text("\n".join(json.dumps(r) for r in rows_out))
print(f"\n[+] Per-query results → {out_path}")

# ── Save summary JSON ──────────────────────────────────────────────────────────

summary = {
    "model":       args.model,
    "qrels":       args.qrels_path,
    "tag":         args.tag,
    "corpus_size": len(corpus_ids),
    "n_queries":   len(query_ids),
    "metrics": {
        metric: round(mean_metric(metric), 4)
        for metric in ordered
    },
    "per_author_ndcg10": {
        author: round(float(np.mean(vals)), 4)
        for author, vals in author_scores.items()
    } if snippet_author else {},
    "per_author_subfield_ndcg10": {
        f"{a} | {sf}": round(float(np.mean(vals)), 4)
        for (a, sf), vals in pair_ndcg10.items()
    } if snippet_subfield else {},
}

summary_path = bench_dir / f"qwen3_{model_slug}{tag_suffix}{cache_suffix}_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
print(f"[+] Summary           → {summary_path}")
