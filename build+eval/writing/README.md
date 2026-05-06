# Writing-Style: Analogues via Cross-Domain Authorship

This task evaluates retrieval systems on finding prose written by the **same author** when topic is deliberately varied—testing whether stylistic invariants persist across unrelated subjects.

## Task Description

- **Corpus**: 10K snippets from 64 researchers + length-matched distractors
- **Queries**: 512 snippets (each gold snippet serves as a query)
- **Relevance**: Other snippets by the same author (authorship = ground truth)
- **Challenge**: Topic similarity is orthogonal to relevance; only stylistic fingerprints matter

## Design Principles

1. **Cross-topic authorship**: Snippets from the same author span unrelated subjects
2. **No easy shortcuts**: Author names and bylines are scrubbed
3. **Same-post exclusion**: Snippets from the same source post are masked at retrieval time
4. **Register-matched distractors**: From LWN.net, LessWrong, and Quanta Magazine

## Pipeline Overview (Paper §4, Appendix C.4)

Unlike other tasks, this skips annotation and clustering because **authorship provides ground truth**:

```
Stage 1: Define Lens      → "authorship / writing style"
Stage 2-3: Skip           → Authorship is the external gold label
Stage 4: Query = Snippet  → Each gold snippet is a query; same-author snippets are positives
Stage 5: No pooling       → Relevance determined by authorship, not post-hoc judgment
```

## Files

### Corpus Building

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `build_benchmark_v2.py` | Assembles corpus from gold track CSV + distractor track CSV + snippet files. Outputs BEIR format with per-query exclusion lists. | §4 (Writing-Style) |
| `scrape_lesswrong.py` | Scrapes LessWrong posts for distractor snippets. | §4 |
| `scrape_lwn_guest.py` | Scrapes LWN.net guest articles for distractor snippets. | §4 |
| `scrape_quanta.py` | Scrapes Quanta Magazine articles for distractor snippets. | §4 |
| `pool_and_judge.py` | Post-retrieval pooling and judgment (for extending annotations if needed). | §4 |
| `build_math_analogues.py` | (Misplaced duplicate from math task) | - |

### Evaluation Scripts

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `eval_gemini_v2.py` | Dense retrieval eval using Gemini-2-Embedding. | §5, Table 3 |
| `eval_qwen3_v2.py` | Dense retrieval eval using Qwen3-Embedding variants. | §5, Table 3 |
| `eval_lateon_v2.py` | Late interaction evaluation using LateOn. | §5, Table 3 |
| `eval_bm25_v2.py` | BM25 lexical baseline. Surprisingly competitive on this task. | §5, Table 3 |
| `eval_gpt_rewriter_gemini_v2.py` | GPT Query Rewriter + Gemini retrieval. **Hurts performance** on this task. | §5, Table 3 |
| `eval_gpt_multihop_writing_v2.py` | GPT Multi-Hop Agent. Also hurts due to topic-preserving reformulation. | §5, Table 3 |
| `eval_gpt5_mergesortinter_v2.py` | Oracle GPT Tournament: stylistic comparison between snippets. | §5, Table 3 |

## Usage

### 1. Prepare Data

Ensure you have:
- `corpus_track.csv` - Gold snippets with author name, post title, etc. (anonymized for privacy concerns)
- `distractor_track.csv` - Distractor snippets (not made public for privacy concerns)
- `corpus/` directory with `{snippet_id}.txt` files

### 2. Build Dataset
```bash
python build_benchmark_v2.py \
    --gold_track corpus_track.csv \
    --distractor_track distractor_track.csv \
    --corpus_dir corpus/ \
    --out_dir benchmark_v2/ \
    --min_per_author 6
```

### 3. Run Evaluation
```bash
# Dense retrieval
export GEMINI_API_KEY="..."
python eval_gemini_v2.py

# BM25 (surprisingly strong on this task)
python eval_bm25_v2.py
```

## Key Results (Paper Table 3)

| System | NDCG@10 | Recall@10 |
|--------|---------|-----------|
| BM25 | .077 | .062 |
| LateOn 0.1B | **.105** | .087 |
| Gemini-2-Embedding | .164 | .132 |
| GPT Query Rewriter | .018 | .008 |
| GPT Multi-Hop Agent | .061 | .034 |
| Oracle GPT Tournament | **.515** | **.449** |

### Key Insights

1. **BM25 beats Qwen embeddings**: Authorial style leaves subtle lexical residue even across topics
2. **LateOn outperforms its size**: Token-level matching captures stylistic patterns
3. **GPT rewriting hurts substantially**: Reformulating prose into keywords preserves topic while **destroying the stylistic signal** that determines relevance
4. **Tournament reads side-by-side**: Achieves .515 NDCG@10 by comparing query and candidate directly

## Output Format

- `corpus.jsonl` - All snippets
- `queries.jsonl` - Snippets from authors with ≥ min per author gold snippets
- `qrels.tsv` - Binary relevance (1 = same author)
- `per_query_excluded_ids.json` - Same-post snippets to exclude at retrieval time
