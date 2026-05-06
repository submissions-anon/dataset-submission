"""
eval_qwen3_congress.py
=======================
Qwen3-Embedding dense retrieval for the Congressional Hearing ToT benchmark.

Reads BEIR format directly — no separate txt files needed.

Requires:
  pip install transformers torch pytrec_eval tqdm numpy

Usage:
  # 0.6B
  python eval_qwen3_congress.py \
      --benchmark_dir congress_corpus_data/beir_export/ \
      --model Qwen/Qwen3-Embedding-0.6B

  # 4B (needs ~20GB VRAM)
  python eval_qwen3_congress.py \
      --benchmark_dir congress_corpus_data/beir_export/ \
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
                    help="Directory containing corpus.jsonl, queries.jsonl, qrels.tsv")
parser.add_argument("--tag",              default="tot")
parser.add_argument("--model",            default="Qwen/Qwen3-Embedding-0.6B")
parser.add_argument("--use_instruction",  action="store_true",
                    help="Prepend instruction prefix to queries")
parser.add_argument("--batch_size",       type=int, default=16)
parser.add_argument("--max_length",       type=int, default=8192)
parser.add_argument("--top_k",           type=int, default=1000)
parser.add_argument("--chunk_size",       type=int, default=500)
parser.add_argument("--k_values",         default="10,50,100,1000")
parser.add_argument("--ckpt_dir",         default=None)
parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

k_values  = [int(k) for k in args.k_values.split(",")]
bench_dir = Path(args.benchmark_dir)
ckpt_dir  = Path(args.ckpt_dir) if args.ckpt_dir else bench_dir / "qwen3_cache"
ckpt_dir.mkdir(exist_ok=True, parents=True)
model_slug = args.model.split("/")[-1]
cache_suffix = "_instr" if args.use_instruction else "_noinstr"

QUERY_INSTRUCTION = (
    "Given a vague description of a congressional hearing moment, "
    "retrieve the specific hearing transcript passage being described."
)

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

# Query metadata for breakdowns
query_witness = {}
query_memorability = {}
for qid, q in queries.items():
    meta = q.get('metadata', {})
    query_witness[qid] = meta.get('source_speaker', 'unknown').lower()
    query_memorability[qid] = meta.get('memorability', 0)

corpus_ids   = list(corpus.keys())
corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
cid_to_idx   = {cid: i for i, cid in enumerate(corpus_ids)}
query_ids    = list(queries.keys())
query_texts  = [queries[qid]['text'] for qid in query_ids]

print(f"  Corpus:  {len(corpus_ids)} passages")
print(f"  Queries: {len(query_ids)}")

# ── Model ─────────────────────────────────────────────────────────────────────

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
        p = ckpt_dir / f"{ckpt_prefix}_{model_slug}{cache_suffix}_chunk{chunk_start}.npy"
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
        chunk_path = ckpt_dir / f"{ckpt_prefix}_{model_slug}{cache_suffix}_chunk{chunk_start}.npy"
        np.save(chunk_path, chunk_arr)
        out_embs.append(chunk_arr)
        torch.cuda.empty_cache()

    return np.concatenate(out_embs, axis=0)

# ── Encode ────────────────────────────────────────────────────────────────────

print("\n[3/5] Encoding corpus...")
corp_embs = encode_with_checkpoint(corpus_texts, "corpus")

if corp_embs.shape[0] != len(corpus_ids):
    raise SystemExit(
        f"\n[error] Corpus cache has {corp_embs.shape[0]} rows but corpus has {len(corpus_ids)}.\n"
        f"Fix: rm {ckpt_dir}/corpus_*.npy  and rerun."
    )

print(f"\n[4/5] Encoding queries... (use_instruction={args.use_instruction})")
query_cache_prefix = f"queries{cache_suffix}"
query_embs = encode_with_checkpoint(
    query_texts,
    query_cache_prefix,
    instruction=QUERY_INSTRUCTION if args.use_instruction else None,
)

if query_embs.shape[0] != len(query_ids):
    raise SystemExit(
        f"\n[error] Query cache has {query_embs.shape[0]} rows but have {len(query_ids)} queries.\n"
        f"Fix: rm {ckpt_dir}/{query_cache_prefix}_*.npy  and rerun."
    )

# ── Retrieve ──────────────────────────────────────────────────────────────────

print(f"\n[5/5] Retrieving (top_k={args.top_k})...")
SCORE_BATCH = 256
run = {}
ranked_lists = {}

for i in tqdm(range(0, len(query_ids), SCORE_BATCH), desc="Scoring"):
    q_batch = query_embs[i: i + SCORE_BATCH]
    scores  = q_batch @ corp_embs.T
    for j, qid in enumerate(query_ids[i: i + SCORE_BATCH]):
        row     = scores[j]
        top_idx = np.argsort(row)[::-1][: args.top_k]
        run[qid] = {corpus_ids[k]: float(row[k]) for k in top_idx}
        ranked_lists[qid] = [corpus_ids[k] for k in top_idx]

# ── Evaluate ──────────────────────────────────────────────────────────────────

metrics = set()
for k in k_values:
    metrics.add(f"ndcg_cut_{k}")
    metrics.add(f"recall_{k}")
metrics.add("map")

evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
per_query = evaluator.evaluate(run)

ordered = (
    [f"ndcg_cut_{k}" for k in k_values] +
    [f"recall_{k}"   for k in k_values] +
    ["map"]
)

agg = defaultdict(list)
for qid, ms in per_query.items():
    for m, v in ms.items():
        agg[m].append(v)

print(f"\n{'='*60}")
print(f"  Qwen3 Embedding — Congressional Hearing ToT")
print(f"{'='*60}")
print(f"  Model:   {args.model}")
print(f"  Instr:   {args.use_instruction}")
print(f"  Corpus:  {len(corpus_ids)} passages")
print(f"  Queries: {len(query_ids)}")
print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'Std':>8}")
print(f"  {'-'*52}")
for m in ordered:
    if m in agg:
        vals = agg[m]
        print(f"  {m:<25} {np.mean(vals):>8.4f} {np.median(vals):>8.4f} {np.std(vals):>8.4f}")

print(f"\n  Evaluated on {len(per_query)} queries")

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

# ── Save outputs ──────────────────────────────────────────────────────────────

tag_suffix = f"_{args.tag}" if args.tag else ""
results_path = bench_dir / f"qwen3_{model_slug}{tag_suffix}_results.jsonl"
rows_out = []
for qid in query_ids:
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

summary = {
    "model":       args.model,
    "tag":         args.tag,
    "use_instruction": args.use_instruction,
    "corpus_size": len(corpus_ids),
    "n_queries":   len(per_query),
    "metrics":     {m: round(float(np.mean(agg.get(m, [0]))), 4) for m in ordered},
    "per_memorability_ndcg10": {
        str(m): round(float(np.mean(v)), 4) for m, v in sorted(mem_ndcg10.items(), reverse=True)
    },
    "per_witness_ndcg10": {
        w: round(float(np.mean(v)), 4) for w, v in witness_ndcg10.items()
    },
}
summary_path = bench_dir / f"qwen3_{model_slug}{tag_suffix}_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults written:")
print(f"  {results_path}")
print(f"  {summary_path}")
