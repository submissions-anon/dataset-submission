# Twitter-Conflict: Descriptive Retrieval over Implicit Stance

This task evaluates retrieval systems on finding tweets that indicate a given political stance **implicitly**—through sarcasm, irony, hedging, selective framing, or rhetorical questions—rather than stating it directly.

## Task Description

- **Corpus**: 72K English tweets about a geopolitical conflict (collected Feb 2026)
- **Queries**: 281 queries describing abstract stances
- **Relevance**: Only tweets with implicit stances can be gold; explicit statements and news reposts serve as hard distractors
- **Challenge**: Query-document vocabulary overlap is minimal; relevance requires inferring implied meaning

## Pipeline Overview (Paper §4, Appendix C.1)

```
Stage 1: Define Lens      → "implicit political stance"
Stage 2: Annotation       → GPT classifies tweets as explicit/news/implicit; extracts implicit meaning
Stage 3: Clustering       → Theme labels assigned and collapsed into canonical themes
Stage 4: Query Generation → Abstract query per theme (avoiding source vocabulary)
Stage 5: Pool & Expand    → Post-retrieval judgment expansion
```

## Files

### Corpus Building

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `collect_tweets.py` | Collects tweets via X API v2 full-archive search endpoint. Filters by keyword clusters related to the conflict, excludes retweets, English only. | §4 (Twitter-Conflict), Table 1 |
| `build_retrieval_dataset.py` | Full construction pipeline: (1) extracts implicit meaning per tweet, (2) assigns/collapses theme labels, (3) generates queries + relevance grades, (4) outputs BEIR format, (5) expands judgments via embedding-space neighbors. | Figure 4, §4, Appendix C.1 |

### Evaluation Scripts

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `eval_twitter_gemini.py` | Dense retrieval eval using Gemini-2-Embedding or Qwen3-Embedding. Supports corpus modes (implicit-only vs full) and embedding surface (tweet text vs implicit meaning descriptions). | §5, Table 2 |
| `eval_qwen3_twitter.py` | Evaluation with Qwen3-Embedding models (0.6B and 4B variants). | §5, Table 2 |
| `eval_twitter_lateon.py` | Late interaction evaluation using LateOn (149M parameters) with PyLate/PLAID indexing. | §5, Table 2 |
| `eval_gpt_rewriter_gemini.py` | GPT Query Rewriter: single-hop query reformulation followed by Gemini dense retrieval. | §5, Table 2 |
| `eval_gpt_multihop_twitter.py` | GPT Multi-Hop Agent: 4-hop iterative retrieval where GPT generates queries, reads results, and accumulates notes across hops. | §5, Table 2 |
| `eval_gpt_mergesortinter_twitter.py` | Oracle GPT Tournament: listwise reranking over pooled candidates to establish upper-bound verification score. | §3, §5, Appendix B |
| `eval_gpt_mergesortinter_twitter_recursive.py` | Recursive variant of the tournament reranking algorithm. | §3, Appendix B |

## Usage

### 1. Collect Tweets
```bash
export BEARER_TOKEN="your_x_api_bearer_token"
python collect_tweets.py
```

### 2. Build Dataset
```bash
export OPENAI_API_KEY="..."
python build_retrieval_dataset.py
```

Outputs to `dataset/`:
- `corpus_implicit.jsonl` - 7,918 implicit tweets
- `corpus_full.jsonl` - All 72K tweets
- `queries.jsonl` - Retrieval queries
- `qrels.tsv` - Relevance judgments

### 3. Run Evaluation
```bash
# Gemini embedding
export GEMINI_API_KEY="..."
python eval_twitter_gemini.py --provider gemini --corpus full

# Qwen3 embedding
python eval_qwen3_twitter.py --model Qwen/Qwen3-Embedding-4B

# Multi-hop agent
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
python eval_gpt_multihop_twitter.py --num-hops 4
```

## Key Results (Paper Table 2)

| System | NDCG@10 (Pooled) | Recall@10 (Pooled) |
|--------|------------------|---------------------|
| BM25 | .002 | .002 |
| Gemini-2-Embedding | .132 | .100 |
| GPT Multi-Hop Agent | .215 | .176 |
| Oracle GPT Tournament | .436 | .342 |

The large gap between retrieval systems and the oracle reranker demonstrates the retrieval-verification asymmetry central to oblique queries.
