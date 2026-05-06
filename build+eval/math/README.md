# Math Meta-Program: Analogues via Shared Reasoning Technique

This task evaluates retrieval systems on finding math problems that share the same **abstract proof strategy** (the "meta-program" or "aha moment") despite differing in mathematical field and surface topic.

## Task Description

- **Corpus**: 3.5K problems from Putnam, undergraduate competitions, AMM, and qualifying exams
- **Queries**: 151 problems with identified reasoning fingerprints
- **Relevance**: Other problems requiring the same abstract meta-move
- **Challenge**: Relevance depends on the (typically latent) proof structure, not topic similarity

## Example

A query about solving a differential equation using characteristic equations should match:
- A recurrence relation solved via generating functions (same eigenvalue decomposition meta-move)
- An optimization problem using Lagrange multipliers (same constraint projection pattern)

...even though these problems have zero vocabulary overlap.

## Pipeline Overview (Paper §4, Appendix C.3)

```
Stage 1: Define Lens        → "proof strategy / meta-program"
Stage 2: Fingerprinting     → GPT extracts meta_strategy, abstract_proof_move, key_insight
Stage 3: Clustering         → Label normalization + merge identification across clusters
Stage 3b: Validation        → Cluster membership verification
Stage 4: Query Generation   → Problems serve as queries; cluster members as positives
Stage 5: Pool & Expand      → Post-retrieval judgment based on shared reasoning
```

## Files

### Corpus Building

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `build_math_analogues.py` | Full construction pipeline: (1) reasoning fingerprint extraction per problem, (2) meta-program label normalization, (3) cluster validation, (4) diversity-based cluster selection, (5) BEIR export. | Figure 4, §4, Appendix C.3 |

### Evaluation Scripts

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `eval_math_gemini_qwen_analogues.py` | Dense retrieval eval using Gemini-2-Embedding and Qwen3-Embedding variants. | §5, Table 3 |
| `eval_math_analogues_lateon.py` | Late interaction evaluation using LateOn. | §5, Table 3 |
| `eval_gpt_rewriter_gemini.py` | GPT Query Rewriter + Gemini retrieval. | §5, Table 3 |
| `eval_gpt_multihop_math.py` | GPT Multi-Hop Agent: iterative problem-to-problem matching. | §5, Table 3 |
| `eval_gpt_mergesortinter_with_solutions.py` | Oracle GPT Tournament **with solutions**: Uses solution text to identify proof structure. This is the "+Soln" variant in Table 3. | §5, Table 3 |
| `eval_gpt_mergesortinter_think_first_recursive.py` | Two-stage ranking: GPT first reasons about each problem's approach, then ranks. | Appendix C.3.6 |

## Usage

### 1. Build Dataset
```bash
export OPENAI_API_KEY="..."
python build_math_analogues.py
```

### 2. Run Evaluation
```bash
# Dense retrieval
export GEMINI_API_KEY="..."
python eval_math_gemini_qwen_analogues.py

# Tournament with solutions (oracle upper bound)
python eval_gpt_mergesortinter_with_solutions.py
```

## Key Results (Paper Table 3)

| System | NDCG@10 (Pooled) | Recall@10 (Pooled) |
|--------|------------------|---------------------|
| BM25 | .029 | .029 |
| Gemini-2-Embedding | .147 | .156 |
| GPT Multi-Hop Agent | .207 | .167 |
| Oracle GPT Tournament | .329 | .300 |
| Oracle GPT Tournament **+Soln** | **.473** | **.417** |

### Key Insight

When the tournament is given solutions ("+Soln"), it rises from .329 to .473 NDCG@10. This confirms that the relevance relation depends on the typically latent proof structure—when that structure is made explicit via solutions, verification becomes much easier.

## Reasoning Fingerprint Schema

Each problem is annotated with:
- `meta_strategy`: The abstract reasoning move (domain-independent)
- `abstract_proof_move`: Logical skeleton (e.g., "assume extremal → local exchange → contradiction")
- `key_insight`: The non-obvious observation that unlocks the problem
- `fingerprint_summary`: ≤20 word clustering key
- `technique_family`: algebra, combinatorics, geometry, number theory, etc.
- `difficulty_tier`: easy, medium, hard
