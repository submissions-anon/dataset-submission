# Congress Hearings: Tip-of-the-Tongue over Transcript Scenarios

This task evaluates retrieval systems on recovering a **single obscure passage** from a lossy, abstract recollection—matching fuzzy memories to specific congressional hearing exchanges.

## Task Description

- **Corpus**: 213K passages from GovInfo (110th–119th Congress) + 10 high-profile tech hearings
- **Queries**: 254 "tip-of-tongue" descriptions written as forum posts
- **Relevance**: Exactly one gold passage per query
- **Challenge**: Queries describe dynamics, emotional register, and rhetorical moves while omitting names, dates, committees, and verbatim transcript phrasing

## Example Query

> "Someone sent me this clip once where this woman in a Senate room starts off weirdly friendly, making some joke about the guy's whole look, and he kind of relaxes for like half a second. Then she just flips and starts pinning him down while he does that 'I can't really speak to that' thing..."

This describes a specific exchange without any searchable keywords.

## Pipeline Overview (Paper §4)

This task skips clustering because each query targets exactly one passage:

```
Stage 1: Define Lens      → "memorable exchange scenario"
Stage 2: Annotation       → Rate passage memorability (1-5); generate ToT post if ≥3
Stage 2b: Diversification → Vary opening styles across 15 patterns to prevent mode collapse
Stage 3: Skip             → One passage per query (no clustering)
Stage 4: Query Generation → Already done in Stage 2 (recollection as query)
Stage 5: Hardening        → Remove residual identifiers from queries
```

## Files

### Corpus Building

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `build_congress_corpus.py` | Multi-phase corpus construction: (1) list hearing packages from GovInfo API, (2) download HTML + metadata, (3) segment into speaker-turn passages, (4) export to BEIR format. Resume-safe with checkpoints. | §4 (Congress Hearings), Appendix C.5 |

### Evaluation Scripts

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `eval_gemini_congress.py` | Dense retrieval eval using Gemini-2-Embedding. | §5, Table 4 |
| `eval_qwen3_congress.py` | Dense retrieval eval using Qwen3-Embedding variants. | §5, Table 4 |
| `eval_lateon_congress.py` | Late interaction evaluation using LateOn. **Best single-stage retriever** on this task. | §5, Table 4 |
| `eval_bm25_congress.py` | BM25 lexical baseline. Near-zero performance. | §5, Table 4 |
| `eval_gpt_rewriter_gemini_congress.py` | GPT Query Rewriter: translates recollection into transcript-like language. | §5, Table 4 |
| `eval_gpt_multihop_congress.py` | GPT Multi-Hop Agent: iterative narrowing of search. | §5, Table 4 |
| `eval_gpt_mergesort_congress.py` | Oracle GPT Tournament: scenario matching over pooled candidates. | §5, Table 4 |

## Usage

### 1. Build Corpus

```bash
export GOVINFO_API_KEY="your_key_from_api.data.gov"

# Phase 1: List available hearings
python build_congress_corpus.py --phase list

# Phase 2: Download HTML + metadata
python build_congress_corpus.py --phase download

# Phase 3: Segment into passages
python build_congress_corpus.py --phase segment

# Phase 4: Export to BEIR format
python build_congress_corpus.py --phase export

# Or run all phases:
python build_congress_corpus.py --phase all
```

### 2. Run Evaluation
```bash
# Dense retrieval
export GEMINI_API_KEY="..."
python eval_gemini_congress.py

# LateOn (best single-stage)
python eval_lateon_congress.py

# Multi-hop agent
export OPENAI_API_KEY="..."
python eval_gpt_multihop_congress.py
```

## Key Results (Paper Table 4)

| System | NDCG@10 | Recall@10 | Recall@100 |
|--------|---------|-----------|------------|
| BM25 | .000 | .000 | .016 |
| Gemini-2-Embedding | .059 | .079 | .126 |
| **LateOn 0.1B** | **.083** | **.102** | .185 |
| GPT Multi-Hop Agent | .183 | .185 | .185 |
| Oracle GPT Tournament | **.913** | **.957** | **1.00** |

### Key Insights

1. **Starkest retrieval-verification asymmetry**: BM25 never finds the target at rank 10; oracle achieves .913 NDCG@10

2. **LateOn punches above its weight**: A 149M-parameter late interaction model outperforms every other single-stage retriever including Gemini-2-Embedding

3. **Multi-hop plateaus**: Recall@10, @50, and @100 are all .185 for the agent—it either finds the target during one of its hops or never does

4. **Token-level matching helps**: Multi-vector late interaction captures local alignment between transcript and abstract scenario description

## Corpus Segmentation

Passages are segmented at speaker-turn boundaries:

```python
SPEAKER_PATTERNS = [
    "Senator WARREN. ...",
    "Mr. ZUCKERBERG. ...",
    "The CHAIRMAN. ...",
]
```

Exchanges are grouped to preserve Q&A coherence:
- Questioner (Senator/Rep) + respondent (witness) kept together
- Long passages (>1000 words) split at sentence boundaries (~500 word chunks)
- Very short passages (<50 words) filtered as procedural noise

## Opening Style Diversification

To prevent mode collapse in query formulation, 15 distinct opening styles are used:

1. Frustrated question ("This has been driving me crazy...")
2. Mid-thought ("Ok so there was this hearing where...")
3. Setting a scene ("I was at my desk...")
4. Comparison ("It was kind of like that other time...")
5. Challenge to reader ("Does anyone else remember...")
6. ... and 10 more patterns

Each memorable passage generates a recollection using one randomly assigned style.
