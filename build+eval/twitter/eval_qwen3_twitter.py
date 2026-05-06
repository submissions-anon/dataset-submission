"""
eval_qwen3_twitter.py
=====================
Qwen3-Embedding dense retrieval eval for the Twitter Descriptive-IR benchmark.
Evaluates Qwen3-Embedding-0.6B and/or 4B.

Saves full ranked list per query for NDCG-Pooled downstream.

Requirements:
  pip install transformers torch pytrec_eval tqdm numpy

Usage:
  # Both models
  python eval_qwen3_twitter.py

  # Single model
  python eval_qwen3_twitter.py --model Qwen/Qwen3-Embedding-0.6B

  # Full corpus
  python eval_qwen3_twitter.py --corpus full
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import pytrec_eval
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", default="dataset")
parser.add_argument("--model",       default=None,
                    help="Single model. Omit to run both 0.6B and 4B.")
parser.add_argument("--corpus",      default="full", choices=["implicit", "full"])
parser.add_argument("--batch-size",  type=int, default=64)
parser.add_argument("--max-length",  type=int, default=999999)
parser.add_argument("--top-k",       type=int, default=1000)
parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--ckpt-dir",    default="eval_cache_qwen3")
args = parser.parse_args()

MODELS = (
    [args.model] if args.model
    else ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B"]
)
QUERY_INSTRUCTION = "Given a query, retrieve relevant tweets that match the implied stance or meaning of the query"

dataset_dir = Path(args.dataset_dir)
ckpt_dir    = Path(args.ckpt_dir)
ckpt_dir.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, tid, score = line.split("\t")
            qrels.setdefault(qid, {})[tid] = int(score)
    return qrels

print(f"[*] Corpus  : {args.corpus}")
print(f"[*] Device  : {args.device}")

corpus_data = load_jsonl(dataset_dir / f"corpus_{args.corpus}.jsonl")
queries_data = load_jsonl(dataset_dir / "queries_merged.jsonl")
qrels        = load_qrels(dataset_dir / "qrels_merged.tsv")

doc_ids   = [d["_id"]  for d in corpus_data]
doc_texts = [d["text"] for d in corpus_data]

query_ids   = [q["_id"]  for q in queries_data if q["_id"] in qrels]
query_texts = [q["text"] for q in queries_data if q["_id"] in qrels]

avg_rel = np.mean([len(v) for v in qrels.values()])
print(f"[*] Corpus docs : {len(doc_ids):,}")
print(f"[*] Queries     : {len(query_ids)}")
print(f"[*] Avg relevant: {avg_rel:.1f}")

# ── Encoding ──────────────────────────────────────────────────────────────────

def last_token_pool(hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[
        torch.arange(hidden_states.shape[0], device=hidden_states.device),
        seq_lengths,
    ]

def encode(texts, tokenizer, model, desc="Encoding"):
    all_embs = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc=f"  {desc}"):
        batch = texts[i:i+args.batch_size]
        enc   = tokenizer(
            batch, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        ).to(args.device)
        with torch.no_grad():
            out = model(**enc)
        emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().float())
    return torch.cat(all_embs, dim=0).numpy()

def encode_with_cache(texts, tokenizer, model, cache_path, desc="Encoding"):
    if cache_path.exists():
        print(f"  Loaded from cache: {cache_path}")
        return np.load(cache_path)
    embs = encode(texts, tokenizer, model, desc=desc)
    np.save(cache_path, embs)
    print(f"  Saved to {cache_path}")
    return embs

# ── Eval loop ─────────────────────────────────────────────────────────────────

all_results = {}

for model_name in MODELS:
    print(f"\n{'='*65}")
    print(f"  Model: {model_name}")
    print(f"{'='*65}")

    model_slug = model_name.replace("/", "_").replace("-", "_")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(args.device).eval()

    # Encode corpus
    doc_cache = ckpt_dir / f"corpus_{model_slug}_{args.corpus}.npy"
    doc_embs  = encode_with_cache(doc_texts, tokenizer, model, doc_cache, "Corpus")

    # Encode queries with instruction prefix
    formatted_queries = [
        f"Instruct: {QUERY_INSTRUCTION}\nQuery: {t}"
        for t in query_texts
    ]
    q_cache    = ckpt_dir / f"queries_{model_slug}_{args.corpus}.npy"
    query_embs = encode_with_cache(formatted_queries, tokenizer, model, q_cache, "Queries")

    # Retrieve
    print(f"  Retrieving top-{args.top_k}...")
    SCORE_BATCH = 256
    run = {}
    for i in tqdm(range(0, len(query_ids), SCORE_BATCH), desc="  Scoring"):
        q_batch   = query_embs[i:i+SCORE_BATCH]
        scores_np = q_batch @ doc_embs.T
        for j, qid in enumerate(query_ids[i:i+SCORE_BATCH]):
            row     = scores_np[j]
            top_idx = np.argsort(row)[::-1][:args.top_k]
            run[qid] = {doc_ids[k]: float(row[k]) for k in top_idx}

    # Evaluate
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

    best_ranks = []
    for qid in query_ids:
        rel_docs = {tid for tid, s in qrels[qid].items() if s >= 1}
        ranked   = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
        rank_map = {doc: r+1 for r, (doc, _) in enumerate(ranked)}
        ranks    = [rank_map[d] for d in rel_docs if d in rank_map]
        if ranks:
            best_ranks.append(min(ranks))
    best_ranks_arr = np.array(best_ranks)

    print(f"\n  {'Metric':<22} {'Score':>8}")
    print(f"  {'-'*32}")
    print(f"  {'MRR':<22} {mean('recip_rank'):>8.4f}")
    print(f"  {'NDCG@10':<22} {mean('ndcg_cut_10'):>8.4f}  <- primary")
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

    # Save per-query results (with full ranked list for NDCG-Pooled)
    out_name = f"results_qwen3_{model_slug}_{args.corpus}_full_final.jsonl"
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
            "best_rank":  min(ranks) if ranks else None,
            "n_relevant": len(qrels[qid]),
            "mrr":        round(v.get("recip_rank",   0), 4),
            "ndcg@10":    round(v.get("ndcg_cut_10",  0), 4),
            "ndcg@50":    round(v.get("ndcg_cut_50",  0), 4),
            "ndcg@100":   round(v.get("ndcg_cut_100", 0), 4),
            "recall@10":  round(v.get("recall_10",    0), 4),
            "recall@50":  round(v.get("recall_50",    0), 4),
            "recall@100": round(v.get("recall_100",   0), 4),
            "success@1":  round(v.get("success_1",    0), 4),
            "success@5":  round(v.get("success_5",    0), 4),
            "success@10": round(v.get("success_10",   0), 4),
            "ranked":     sorted(run[qid].items(), key=lambda x: -x[1])[:args.top_k],
        })

    out_path.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\n[+] Per-query results -> {out_path}")

    all_results[model_name] = mean("ndcg_cut_10")

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  Summary — corpus: {args.corpus}")
print(f"{'='*65}")
print(f"  {'Model':<40} {'NDCG@10':>8}")
print(f"  {'-'*50}")
for mn, ndcg in all_results.items():
    print(f"  {mn.split('/')[-1]:<40} {ndcg:>8.4f}")
print(f"{'='*65}")
