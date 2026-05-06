# Retrieval Benchmark: Latent and Implicit Queries

This repository contains the build and evaluation scripts for a retrieval benchmark suite that exposes overlooked bottlenecks in modern retrievers through latent and implicit queries.

## Repository Structure

```
dataset-submission/
├── README.md                    # This file
├── build+eval/                  # Build and evaluation scripts
│   ├── twitter/                 # Twitter-Conflict: Descriptive retrieval over implicit stance
│   ├── wild/                    # WildChat Errors: Descriptive retrieval over LLM failure modes
│   ├── math/                    # Math Meta-Program: Analogues via shared reasoning technique
│   ├── writing/                 # Writing-Style: Analogues via cross-domain authorship
│   └── congress/                # Congress Hearings: Tip-of-the-tongue over transcript scenarios
```

## Task Overview

The benchmark consists of **five retrieval tasks** spanning three types of oblique search queries:

### Descriptive Queries
These seek documents that express a latent property implicitly:

| Task | Description | Corpus Size | Queries |
|------|-------------|-------------|---------|
| **Twitter-Conflict** | Find tweets indicating a given stance *implicitly* (through irony, hedging, framing) | 72K tweets | 281 |
| **WildChat Errors** | Find Human-AI conversations exhibiting a specific behavioral failure mode | 507K conversations | 40 |

### Analogue Queries
These seek documents sharing an abstract structure with the query:

| Task | Description | Corpus Size | Queries |
|------|-------------|-------------|---------|
| **Math Meta-Program** | Given a math problem, retrieve others requiring the same proof strategy (across different fields) | 3.5K problems | 151 |
| **Writing-Style** | Given a text snippet, retrieve other texts by the same author (across different topics) | 10K snippets | 512 |

### Tip-of-the-Tongue Queries
These match a fuzzy recollection to a specific obscure passage:

| Task | Description | Corpus Size | Queries |
|------|-------------|-------------|---------|
| **Congress Hearings** | Recover a single obscure passage from a lossy, abstract recollection | 213K passages | 254 |

## Evaluated Systems

The scripts evaluate multiple retrieval architectures:

- **Lexical**: BM25 (baseline)
- **Dense Retrievers**: Gemini-2-Embedding, Qwen3-Embedding (0.6B and 4B)
- **Late Interaction**: LateOn (149M parameters)
- **Agentic Pipelines**:
  - GPT Query Rewriter (single-hop reformulation)
  - GPT Multi-Hop Agent (4-hop iterative retrieval)
- **Oracle Reranker**: GPT Tournament (listwise reranking for upper-bound estimation)

## Directory Details

Each subdirectory contains its own README with detailed file descriptions:

- [`build+eval/twitter/README.md`](build+eval/twitter/README.md) - Implicit stance retrieval
- [`build+eval/wild/README.md`](build+eval/wild/README.md) - LLM failure mode retrieval
- [`build+eval/math/README.md`](build+eval/math/README.md) - Mathematical reasoning analogues
- [`build+eval/writing/README.md`](build+eval/writing/README.md) - Cross-domain authorship
- [`build+eval/congress/README.md`](build+eval/congress/README.md) - Tip-of-tongue transcript retrieval

## Requirements

Common dependencies across tasks:
```bash
pip install openai google-genai sentence-transformers pytrec_eval tqdm numpy
```

Task-specific requirements are documented in each subdirectory.

## Environment Variables

```bash
export OPENAI_API_KEY="..."      # For GPT-based pipelines
export GEMINI_API_KEY="..."      # For Gemini embeddings
export BEARER_TOKEN="..."        # For Twitter API (twitter task)
export GOVINFO_API_KEY="..."     # For GovInfo API (congress task)
```

## Output Format

All tasks output data in BEIR-compatible format:
- `corpus.jsonl` - Documents with `_id`, `text`, `metadata`
- `queries.jsonl` - Queries with `_id`, `text`, `metadata`
- `qrels.tsv` - Relevance judgments: `query-id`, `corpus-id`, `score`

## Metrics

Primary metrics reported:
- **NDCG@10, NDCG@50** - Normalized Discounted Cumulative Gain
- **Recall@10, @50, @100** - Fraction of relevant documents retrieved

Both "Gold" (original annotations) and "Pooled" (post-retrieval expanded judgments) evaluations are supported.
