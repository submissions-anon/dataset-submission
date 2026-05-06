# WildChat Errors: Descriptive Retrieval over LLM Failure Modes

This task evaluates retrieval systems on finding Human-AI conversations that exhibit a specific **behavioral failure mode**—where the failure has no explicit lexical marker in the text.

## Task Description

- **Corpus**: 507K conversations from WildChat-4.8M (2025 subset, English)
- **Queries**: 40 queries describing specific failure patterns
- **Relevance**: Conversations flagged with matching failure modes
- **Challenge**: Failure modes are distributed across turns; no single phrase signals the error

## Example Failure Modes

- AI's output contains visible formatting corruption the user didn't request, and the AI fails to self-correct
- Model silently drops part of the user's instruction without acknowledgment
- AI makes an unjustified unit conversion or format change

## Pipeline Overview (Paper §4, Appendix C.2)

```
Stage 1: Define Lens      → "LLM behavioral failure mode"
Stage 2: Annotation       → GPT-5.4-nano corpus-wide sweep flags 2.9% of conversations
Stage 3: Clustering       → Embedding-based clustering + LLM consolidation into canonical types
Stage 4: Query Generation → Abstract failure description per cluster
Stage 5: Pool & Expand    → Post-retrieval judgment expansion with conservative matching
```

## Files

### Corpus Building

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `get_wildchat_2025.py` | Downloads and filters WildChat-4.8M to 2025 English conversations. Serializes as alternating user-assistant turns. | §4 (WildChat Errors) |
| `get_wildchat_2025_chunked.py` | Chunked variant for large corpus processing. | §4 |
| `build_wildchat_retrieval.py` | Full construction pipeline: (1) corpus-wide failure detection, (2) label clustering/consolidation, (3) query generation, (4) relevance judging, (5) BEIR export. | Figure 4, §4, Appendix C.2 |

### Evaluation Scripts

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `eval_wildchat_gemini.py` | Dense retrieval eval using Gemini-2-Embedding. | §5, Table 2 |
| `eval_wildchat_lateon.py` | Late interaction evaluation using LateOn. | §5, Table 2 |
| `bm25_eval_wildchat.py` | BM25 lexical baseline evaluation. | §5, Table 2 |
| `eval_gpt_rewriter_gemini.py` | GPT Query Rewriter: single-hop reformulation + Gemini retrieval. | §5, Table 2 |
| `eval_gpt_multihop_wildchat.py` | GPT Multi-Hop Agent: 4-hop iterative retrieval for failure mode search. | §5, Table 2 |
| `eval_gpt_mergesortinter_wildchat.py` | Oracle GPT Tournament: listwise reranking for upper-bound estimation. | §3, §5, Appendix B |
| `eval_gpt_mergesortinter_wildchat_recursive.py` | Recursive variant of tournament reranking. | Appendix B |

## Usage

### 1. Prepare Corpus
```bash
python get_wildchat_2025.py
# or for chunked processing:
python get_wildchat_2025_chunked.py
```

### 2. Build Dataset
```bash
export OPENAI_API_KEY="..."
python build_wildchat_retrieval.py
```

### 3. Run Evaluation
```bash
# Dense retrieval
export GEMINI_API_KEY="..."
python eval_wildchat_gemini.py

# Multi-hop agent
export OPENAI_API_KEY="..."
python eval_gpt_multihop_wildchat.py
```

## Key Results (Paper Table 2)

| System | NDCG@10 (Pooled) | Recall@10 (Pooled) |
|--------|------------------|---------------------|
| BM25 | .006 | .005 |
| Gemini-2-Embedding | .097 | .075 |
| GPT Multi-Hop Agent | .113 | .077 |
| Oracle GPT Tournament | .431 | .308 |

### Key Insight

Multi-hop search helps WildChat less than Twitter because conversational failures are distributed across many turns, making them harder to approach through iterative query reformulation. The oracle reranker achieves strong performance by reading query and conversation together.
