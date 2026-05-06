"""
build_math_analogues.py
-----------------------
Builds a BEIR-style reasoning-analogue retrieval benchmark from a math problem dataset.

Input:  final_dataset.json  (fields: competition, year, problem_id, problem, solution)
Output: BEIR-format dataset — corpus.jsonl, queries.jsonl, qrels.tsv

Pipeline:
  Phase 1 — Fingerprint : LLM extracts reasoning fingerprint per problem
                          (meta_strategy, abstract_proof_move, key_insight,
                           fingerprint_summary). Checkpointed, one call per problem.

  Phase 2 — Cluster (LLM-only):
      2a. Extract fingerprint_summaries (≤20-word abstract labels per problem)
      2b. Three-pass LLM collapse of near-duplicate labels → canonical meta-programs
      2c. Re-map every problem to its canonical meta-program
      2d. LLM merge pass: clusters whose members share the same aha moment merge

  Phase 3 — Query build (programmatic, no LLM):
      One query per problem: problem statement + framing prompt asking for
      same reasoning pattern. Self is excluded from qrels.

  Phase 4 — Qrel assignment:
      Score 2 = co-cluster (same canonical meta-program — same aha moment)

  Phase 5 — Output:
      corpus.jsonl, queries.jsonl, qrels.tsv (BEIR format)
      queries_sample.jsonl: stratified 500-query subset for reporting
      summary.jsonl: human-readable per-query breakdown

Usage:
    pip install openai tqdm
    export OPENAI_API_KEY="sk-..."
    python build_math_analogues.py [--input final_dataset.json] [--out_dir dataset]
"""

import os, json, time, argparse, random
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

MODEL        = "gpt-5"
LABEL_COLLAPSE_CHUNK = 150   # labels per collapse LLM call
MERGE_CHUNK          = 80    # clusters per merge LLM call
MIN_CLUSTER_SIZE     = 2     # minimum cluster size to keep
SAMPLE_SIZE          = 500   # queries in the stratified reporting sample

# ── Client ────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def gpt_json(messages, max_tokens=50000, retries=4):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=1.0,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  [gpt] error attempt {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return None


def composite_id(p: dict) -> str:
    """Stable unique ID: competition__year__problem_id"""
    return f"{p.get('competition','')}___{p.get('year','')}___{p['problem_id']}"


def load_problems(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    raw = data if isinstance(data, list) else next(
        (v for v in data.values() if isinstance(v, list)), None
    )
    if raw is None:
        raise ValueError(f"Cannot parse problems list from {path}")
    # Deduplicate on composite key; stamp each problem with its composite _id
    seen, out = set(), []
    for p in raw:
        cid = composite_id(p)
        if cid not in seen:
            seen.add(cid)
            p["_id"] = cid   # attach so all phases use it
            out.append(p)
    n_dupes = len(raw) - len(out)
    if n_dupes:
        print(f"  Deduplicated {n_dupes} duplicate entries "
              f"({len(raw)} rows → {len(out)} unique problems)")
    return out


# ── Phase 1: Fingerprint ───────────────────────────────────────────────────────

FINGERPRINT_SYSTEM = """\
You are a mathematics professor who has spent decades studying how mathematical
reasoning recurs across competitions, fields, and difficulty levels.

Your task: read a problem and its solution(s), then extract a REASONING FINGERPRINT
that captures the abstract cognitive move required, that is, the "aha moment",  stated so
domain-independently that a professor would recognize the same move in a problem
from a completely different mathematical field.

==========================================================================
CRITICAL: GRANULARITY OF meta_strategy AND fingerprint_summary
==========================================================================

These two fields are the PRIMARY CLUSTERING KEYS. They must be abstract enough
that problems from completely different fields using the same reasoning move
land in the same cluster.

CALIBRATION EXAMPLE — three problems sharing ONE meta-program:

  Problem A (Putnam 1995 A-4): Necklace with integer bead labels summing to n-1;
    prove some rotation gives partial sums all ≤ k-1.

  Problem B (Putnam 2017 A-4): 2N students with average score 7.4; prove they
    split into two equal groups each averaging exactly 7.4.

  Problem C (Putnam 2013 A-4): Binary digits on a circle with near-uniform arc
    counts; prove some arc achieves the exact average Z and N.

  CORRECT fingerprint_summary for all three:
    "sum quantity over all rotations or arrangements, evaluate global average
     in closed form, invoke integrality to certify existence of specific instance"

  WRONG (too coarse): "averaging argument" — describes thousands of problems
  WRONG (too fine): "cyclic shift of bead labels forces partial sum bound" — one problem

RULE: the fingerprint_summary must be abstract enough to match a family of
problems, specific enough to exclude unrelated reasoning moves.

TOO COARSE (useless): "induction", "pigeonhole", "contradiction", "combinatorics"
TOO FINE (useless): anything that describes exactly one problem

==========================================================================
OUTPUT FORMAT — respond with ONLY a valid JSON object, no markdown, no commentary
==========================================================================

{
  "meta_strategy":
    "The abstract reasoning move, the eurak or "aha" moment. 
     Use NO domain vocabulary (no 'necklace', 'bead', 'student', 'matrix').
     Must describe the logical maneuver so that a problem from a completely
     different field using the same move would fit this description.",

  "abstract_proof_move":
    "Domain-independent logical skeleton. Examples:
     'sum over all rotations → global average → existence by integrality' |
     'assume extremal object → local exchange argument → contradiction' |
     'monovariant strictly monotone → finiteness gives termination' |
     'embed discrete in continuous → IVT/MVT → round back to discrete'",

  "key_insight":
    "One sentence: the non-obvious observation that unlocks the problem.
     State WITHOUT domain vocabulary.
     BAD: 'notice the sum of bead labels equals n-1'
     GOOD: 'the global sum over all configurations is computable in closed form,
            so some configuration must achieve the target by an averaging argument'",

  "technique_family":
    "ONE of: algebra | combinatorics | geometry | number_theory |
     real_analysis | linear_algebra | probability | game_theory",

  "surface_domain":
    "Brief phrase: what the problem looks like superficially — vocabulary
     a naive keyword search would match on.
     E.g.: necklace labeling, student score distributions, binary sequences on a circle",

  "difficulty_tier":
    "ONE of: easy | medium | hard",

  "fingerprint_summary":
    "≤20 words. The meta-program label for this entire family of problems.
     Must be abstract enough that problems from different fields using the
     same reasoning move land here. PRIMARY CLUSTERING KEY."
}"""


def fingerprint_one(p: dict) -> dict | None:
    soln = (p.get("solution") or "").strip() or \
           "[No solution provided — fingerprint from problem statement only]"
    user_msg = (
        f"PROBLEM ID: {p['_id']}\n\n"
        f"PROBLEM:\n{p['problem']}\n\n"
        f"SOLUTION:\n{soln}\n\n"
        "Produce the reasoning fingerprint JSON now."
    )
    result = gpt_json(
        [{"role": "system", "content": FINGERPRINT_SYSTEM},
         {"role": "user",   "content": user_msg}],
        max_tokens=50000,
    )
    if result:
        result["problem_id"] = p["_id"]
    return result


def phase1_fingerprint(problems: list, ckpt_dir: Path) -> dict:
    """Returns {problem_id -> fingerprint_dict}.
    Always diffs against the full problem list so previously-failed problems
    (those that returned None and were never written to disk) get retried.
    """
    ckpt    = ckpt_dir / "fingerprints.json"
    partial = ckpt_dir / "fingerprints_partial.jsonl"

    # Load whatever we have so far (checkpoint takes priority over partial)
    fps = {}
    if ckpt.exists():
        raw_fps = json.loads(ckpt.read_text())
        # Migrate: if keys are bare problem_ids (old format), remap to composite _id.
        # Detect by checking whether any key contains "___" (composite separator).
        if raw_fps and not any("___" in k for k in list(raw_fps.keys())[:5]):
            pid_to_problem = {p["problem_id"]: p for p in problems}
            for k, fp in raw_fps.items():
                p = pid_to_problem.get(k)
                if p:
                    fp["problem_id"] = p["_id"]  # update stored field too
                    fps[p["_id"]] = fp
                else:
                    fps[k] = fp  # keep as-is if no match
            print(f"[Phase 1] Loaded {len(fps)} fingerprints from checkpoint (migrated keys)")
        else:
            fps = raw_fps
            print(f"[Phase 1] Loaded {len(fps)} fingerprints from checkpoint")
    elif partial.exists():
        with open(partial) as f:
            for line in f:
                line = line.strip()
                if line:
                    fp = json.loads(line)
                    fps[fp["problem_id"]] = fp
        print(f"[Phase 1] Resuming — {len(fps)} already fingerprinted")

    # Always diff — catches problems that previously failed (returned None)
    # and were silently dropped without being written to disk
    todo = [p for p in problems if p["_id"] not in fps]

    if not todo:
        print(f"[Phase 1] All {len(problems)} problems fingerprinted — done")
        return fps

    print(f"[Phase 1] {len(todo)} problems missing fingerprints "
          f"({len(fps)} already done) — fingerprinting via {MODEL}...")

    with open(partial, "a") as pf:
        for p in tqdm(todo, desc="Fingerprinting"):
            fp = fingerprint_one(p)
            if fp:
                fps[p["_id"]] = fp
                pf.write(json.dumps(fp) + "\n")
                pf.flush()
            else:
                print(f"  [WARN] Failed: {p['_id']}")
            time.sleep(0.5)

    ckpt.write_text(json.dumps(fps, indent=2))
    partial.unlink(missing_ok=True)
    n_missing = len(problems) - len(fps)
    print(f"  Fingerprinted: {len(fps)}/{len(problems)}"
          + (f"  ({n_missing} still missing — re-run to retry)" if n_missing else ""))
    return fps


# ── Phase 2: LLM-only Clustering ──────────────────────────────────────────────

COLLAPSE_SYSTEM = """\
You are normalizing a list of meta-program labels from mathematical reasoning analysis.

Many labels express the same underlying reasoning move with slightly different wording.
Collapse near-duplicates into a single canonical label. Keep genuinely distinct
reasoning moves as separate canonicals.

CRITICAL DISTINCTION: two labels should collapse only if a mathematician would say
"yes, these require the same core aha moment" — NOT merely that they use the same
technique family, topic area, or proof technique.

Return ONLY a JSON object mapping every original label to its canonical form:
{"original label": "canonical label", ...}

Rules:
- Every input label must appear exactly once as a key
- Pick the most abstract, clearly-phrased version as the canonical
- Multiple originals can (and should) map to the same canonical
- Do NOT collapse labels that are merely in the same technique family"""


def run_collapse_pass(labels: list) -> dict:
    """One collapse pass over a list of label strings."""
    chunks = [labels[i:i+LABEL_COLLAPSE_CHUNK]
              for i in range(0, len(labels), LABEL_COLLAPSE_CHUNK)]
    merged = {}
    for chunk in tqdm(chunks, desc="  Collapsing", leave=False):
        block = "\n".join(f"- {l}" for l in chunk)
        result = gpt_json(
            [{"role": "system", "content": COLLAPSE_SYSTEM},
             {"role": "user",   "content": block}],
            max_tokens=50000,
        )
        if result and isinstance(result, dict):
            merged.update(result)
        else:
            # identity map on failure — don't lose any labels
            for l in chunk:
                merged[l] = l
        time.sleep(0.4)
    return merged


MERGE_SYSTEM = """\
You are an expert mathematician reviewing clusters of math problems grouped by
their meta-program — the abstract reasoning move required to solve them.

You will receive a list of clusters, each with a canonical label and the
fingerprint_summaries of a sample of its member problems.

Identify which clusters should be MERGED because they represent the same
fundamental reasoning pattern — the same abstract "aha moment" — even if
described with different wording.

DO NOT merge clusters that merely belong to the same technique family
(e.g., "all use induction") but require genuinely different insights.

Return ONLY a JSON object:
{
  "merges": [
    {
      "merge_ids": [<cluster_index_1>, <cluster_index_2>, ...],
      "new_label": "<canonical label for merged cluster — ≤20 words, abstract>",
      "rationale": "<one sentence: why these share the same meta-program>"
    },
    ...
  ]
}

If no merges are warranted, return {"merges": []}.
Only include groups that should actually merge. Each merge group must have ≥2 indices."""


def phase2_cluster(fps: dict, ckpt_dir: Path) -> dict:
    """Returns {canonical_meta_program -> [problem_id, ...]}"""
    ckpt = ckpt_dir / "clusters.json"
    if ckpt.exists():
        print("[Phase 2] Loading clusters from checkpoint...")
        return json.loads(ckpt.read_text())

    # ── 2a: extract fingerprint_summaries as raw labels ────────────────────
    print("[Phase 2a] Extracting fingerprint summaries...")
    pid_to_summary = {
        pid: fp.get("fingerprint_summary", "").strip()
        for pid, fp in fps.items()
    }
    unique_summaries = sorted(set(s for s in pid_to_summary.values() if s))
    print(f"  {len(unique_summaries)} unique fingerprint_summaries "
          f"from {len(fps)} problems")

    # ── 2b: two-pass label collapse ────────────────────────────────────────
    collapse_ckpt = ckpt_dir / "label_collapse.json"
    if collapse_ckpt.exists():
        print("[Phase 2b] Loading collapse map from checkpoint...")
        collapse_map = json.loads(collapse_ckpt.read_text())
    else:
        print(f"[Phase 2b] Collapsing {len(unique_summaries)} labels — pass 1...")
        pass1 = run_collapse_pass(unique_summaries)

        after_pass1 = sorted(set(pass1.values()))
        print(f"  After pass 1: {len(after_pass1)} labels. Running pass 2...")
        pass2 = run_collapse_pass(after_pass1)

        # Pass 3: one final sweep over the pass-2 survivors
        after_pass2 = sorted(set(pass2.values()))
        print(f"  After pass 2: {len(after_pass2)} labels. Running pass 3...")
        pass3 = run_collapse_pass(after_pass2)

        # Chain: raw -> pass1 -> pass2 -> pass3
        collapse_map = {}
        for orig, mid1 in pass1.items():
            mid2 = pass2.get(mid1, mid1)
            collapse_map[orig] = pass3.get(mid2, mid2)
        collapse_ckpt.write_text(json.dumps(collapse_map, indent=2))

    n_canonical = len(set(collapse_map.values()))
    print(f"  {n_canonical} canonical meta-programs after three-pass collapse")

    # ── 2c: re-map problems to canonical cluster ───────────────────────────
    clusters: dict[str, list] = defaultdict(list)
    unclustered = []
    for pid, summary in pid_to_summary.items():
        if not summary:
            unclustered.append(pid)
            continue
        canonical = collapse_map.get(summary, summary)
        clusters[canonical].append(pid)

    # Drop clusters below minimum size
    clusters = {k: v for k, v in clusters.items() if len(v) >= MIN_CLUSTER_SIZE}
    print(f"  {len(clusters)} clusters with ≥{MIN_CLUSTER_SIZE} members "
          f"({len(unclustered)} problems had no fingerprint_summary)")

    # ── 2d: merge pass ─────────────────────────────────────────────────────
    merge_ckpt = ckpt_dir / "merged_clusters.json"
    if merge_ckpt.exists():
        print("[Phase 2d] Loading post-merge clusters from checkpoint...")
        clusters = json.loads(merge_ckpt.read_text())
    else:
        print(f"[Phase 2d] Merge pass over {len(clusters)} clusters...")
        cluster_list = [(label, list(pids)) for label, pids in clusters.items()]

        def run_merge_pass(cl: list) -> list:
            """One merge pass over cl (list of (label, pids)). Returns new list."""
            chunks = [cl[i:i+MERGE_CHUNK] for i in range(0, len(cl), MERGE_CHUNK)]
            # Work on a mutable copy indexed by position in cl
            result = [list(item) for item in cl]   # [[label, pids], ...]
            label_to_pos = {item[0]: i for i, item in enumerate(result)}

            for chunk in tqdm(chunks, desc="  Merge chunk", leave=False):
                lines = []
                for local_i, (label, pids) in enumerate(chunk):
                    # Show up to 4 members with both fingerprint_summary + key_insight
                    member_lines = []
                    for pid in pids[:10]:
                        if pid not in fps:
                            continue
                        fp = fps[pid]
                        summary = fp.get("fingerprint_summary", "")
                        insight = fp.get("key_insight", "")
                        member_lines.append(f'      - "{summary}" | insight: {insight}')
                    member_block = "\n".join(member_lines) if member_lines else "      (none)"
                    lines.append(
                        f"[{local_i}] {label} (n={len(pids)})\n"
                        f"  Members:\n{member_block}"
                    )
                block = "\n\n".join(lines)
                res = gpt_json(
                    [{"role": "system", "content": MERGE_SYSTEM},
                     {"role": "user",   "content": block}],
                    max_tokens=50000,
                )
                if not res:
                    continue

                chunk_labels = [item[0] for item in chunk]
                for merge in res.get("merges", []):
                    ids = merge.get("merge_ids", [])
                    if len(ids) < 2:
                        continue
                    try:
                        canon_label = chunk_labels[ids[0]]
                        canon_pos   = label_to_pos.get(canon_label)
                        if canon_pos is None:
                            continue
                        for other_local in ids[1:]:
                            other_label = chunk_labels[other_local]
                            other_pos   = label_to_pos.get(other_label)
                            if other_pos is None or other_pos == canon_pos:
                                continue
                            # Absorb other into canon
                            result[canon_pos][1].extend(result[other_pos][1])
                            result[other_pos][0] = None   # mark deleted
                        new_label = merge.get("new_label", canon_label)
                        result[canon_pos][0] = new_label
                        label_to_pos[new_label] = canon_pos
                    except (IndexError, TypeError, KeyError):
                        continue
                time.sleep(0.5)

            return [[l, p] for l, p in result if l and p]

        cluster_list = run_merge_pass(cluster_list)

        # One global merge pass if count is now manageable
        if len(cluster_list) <= MERGE_CHUNK * 3:
            print(f"  Global merge pass over {len(cluster_list)} clusters...")
            cluster_list = run_merge_pass(cluster_list)

        clusters = {
            label: pids
            for label, pids in cluster_list
            if len(pids) >= MIN_CLUSTER_SIZE
        }
        print(f"  {len(clusters)} clusters after merge pass")
        merge_ckpt.write_text(json.dumps(clusters, indent=2))

    ckpt.write_text(json.dumps(clusters, indent=2))

    sizes = sorted(len(v) for v in clusters.values())
    print(f"  Cluster stats: n={len(clusters)}  "
          f"min={sizes[0]}  max={sizes[-1]}  "
          f"median={sizes[len(sizes)//2]}  "
          f"avg={sum(sizes)/len(sizes):.1f}")
    return clusters


# ── Phase 3: Query Build (programmatic, no LLM) ───────────────────────────────

QUERY_TEMPLATE = (
    "Given the following mathematical problem:\n\n"
    "{problem}\n\n"
    "Find other mathematical problems that require the same core reasoning "
    "pattern (i.e., insight) even if they come "
    "from completely different areas of mathematics or use entirely different "
    "vocabulary."
)


def phase3_build_queries(problems: list, fps: dict, clusters: dict,
                         ckpt_dir: Path) -> tuple[list, dict]:
    """
    Returns:
      queries         : list of query dicts
      pid_to_query_id : {problem_id -> query_id}
    """
    ckpt = ckpt_dir / "queries.json"
    if ckpt.exists():
        print("[Phase 3] Loading queries from checkpoint...")
        data = json.loads(ckpt.read_text())
        return data["queries"], data["pid_to_query_id"]

    print("[Phase 3] Building queries (programmatic)...")

    pid_to_cluster = {}
    for label, pids in clusters.items():
        for pid in pids:
            pid_to_cluster[pid] = label

    queries = []
    pid_to_query_id = {}

    for i, p in enumerate(problems):
        pid = p["_id"]
        fp  = fps.get(pid, {})
        qid = f"q{i:05d}"
        queries.append({
            "query_id":         qid,
            "problem_id":       pid,
            "query_text":       QUERY_TEMPLATE.format(problem=p["problem"].strip()),
            "meta_program":     pid_to_cluster.get(pid, ""),
            "technique_family": fp.get("technique_family", ""),
            "competition":      p.get("competition", ""),
            "year":             p.get("year", ""),
            "difficulty_tier":  fp.get("difficulty_tier", ""),
        })
        pid_to_query_id[pid] = qid

    ckpt.write_text(json.dumps(
        {"queries": queries, "pid_to_query_id": pid_to_query_id}, indent=2
    ))
    print(f"  Built {len(queries)} queries")
    return queries, pid_to_query_id


# ── Phase 4: Qrel Assignment ───────────────────────────────────────────────────

def phase4_qrels(queries: list, clusters: dict,
                 ckpt_dir: Path) -> dict:
    """
    Returns base_qrels: {query_id -> {problem_id -> score}}
      Score 2 — co-cluster membership (same canonical meta-program).
    Self is always excluded.
    """
    ckpt = ckpt_dir / "base_qrels.json"
    if ckpt.exists():
        print("[Phase 4] Loading base qrels from checkpoint...")
        return json.loads(ckpt.read_text())

    print("[Phase 4] Building base qrels (score-2 cluster membership only)...")

    pid_to_cluster = {}
    for label, pids in clusters.items():
        for pid in pids:
            pid_to_cluster[pid] = label

    base_qrels = {}
    for q in tqdm(queries, desc="Assigning qrels"):
        qid      = q["query_id"]
        self_pid = q["problem_id"]
        meta     = q["meta_program"]

        scored: dict[str, int] = {}

        # Score 2 only: co-cluster members (same aha moment, same meta-program)
        if meta:
            for pid in clusters.get(meta, []):
                if pid != self_pid:
                    scored[pid] = 2

        if scored:
            base_qrels[qid] = scored

    total = sum(len(v) for v in base_qrels.values())
    n_with_gold = len(base_qrels)
    n_no_gold   = len(queries) - n_with_gold
    print(f"  {total:,} score-2 pairs across {n_with_gold} queries")
    print(f"  {n_no_gold} queries have no cluster-mates (singletons or unclustered) "
          f"— NDCG-Pooled judging will expand coverage post-hoc")

    ckpt.write_text(json.dumps(base_qrels, indent=2))
    return base_qrels


# ── Phase 5: BEIR Output ──────────────────────────────────────────────────────

def phase5_write(problems: list, fps: dict, queries: list,
                 base_qrels: dict,
                 out_dir: Path, sample_size: int = SAMPLE_SIZE) -> None:

    print("[Phase 5] Writing BEIR output files...")
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_by_id = {p["problem_id"]: p for p in problems}

    # ── corpus.jsonl ───────────────────────────────────────────────────────
    with open(out_dir / "corpus.jsonl", "w") as f:
        for p in problems:
            pid = p["_id"]
            fp  = fps.get(pid, {})
            f.write(json.dumps({
                "_id":  pid,
                "text": p["problem"],
                "metadata": {
                    "competition":         p.get("competition", ""),
                    "year":                str(p.get("year", "")),
                    "problem_id":          p["problem_id"],
                    "technique_family":    fp.get("technique_family", ""),
                    "difficulty_tier":     fp.get("difficulty_tier", ""),
                    "surface_domain":      fp.get("surface_domain", ""),
                    "fingerprint_summary": fp.get("fingerprint_summary", ""),
                    "meta_strategy":       fp.get("meta_strategy", ""),
                    "abstract_proof_move": fp.get("abstract_proof_move", ""),
                    "key_insight":         fp.get("key_insight", ""),
                },
            }) + "\n")

    # ── queries.jsonl + qrels.tsv + summary.jsonl ──────────────────────────
    total_pairs  = 0
    kept_queries = 0

    with open(out_dir / "queries.jsonl", "w") as qf, \
         open(out_dir / "qrels.tsv",     "w") as rf, \
         open(out_dir / "summary.jsonl", "w") as sf:

        rf.write("query-id\tcorpus-id\tscore\n")

        for q in queries:
            qid      = q["query_id"]
            self_pid = q["problem_id"]

            scored: dict[str, int] = dict(base_qrels.get(qid, {}))

            if not scored:
                continue   # skip queries with no relevant docs

            qf.write(json.dumps({
                "_id":  qid,
                "text": q["query_text"],
                "metadata": {
                    "problem_id":       self_pid,
                    "meta_program":     q["meta_program"],
                    "technique_family": q["technique_family"],
                    "competition":      q["competition"],
                    "year":             str(q["year"]),
                    "difficulty_tier":  q["difficulty_tier"],
                },
            }) + "\n")

            rel_docs = []
            for pid, score in scored.items():
                rf.write(f"{qid}\t{pid}\t{score}\n")
                p = prob_by_id.get(pid, {})
                rel_docs.append({
                    "problem_id":    pid,
                    "score":         score,
                    "competition":   p.get("competition", ""),
                    "problem_text":  p.get("problem", "")[:200],
                    "fingerprint":   fps.get(pid, {}).get("fingerprint_summary", ""),
                })
                total_pairs += 1

            rel_docs.sort(key=lambda x: -x["score"])
            sf.write(json.dumps({
                "query_id":    qid,
                "problem_id":  self_pid,
                "meta_program": q["meta_program"],
                "query":       q["query_text"][:400],
                "n_relevant":  len(rel_docs),
                "n_score2":    sum(1 for d in rel_docs if d["score"] == 2),
                "n_score1":    sum(1 for d in rel_docs if d["score"] == 1),
                "top_10":      rel_docs[:10],
            }) + "\n")
            kept_queries += 1

    # ── Stratified 500-query reporting sample ──────────────────────────────
    # Stratify by technique_family × difficulty_tier
    all_written_queries = []
    with open(out_dir / "queries.jsonl") as f:
        for line in f:
            all_written_queries.append(json.loads(line))

    strat: dict[tuple, list] = defaultdict(list)
    for q in all_written_queries:
        key = (q["metadata"]["technique_family"],
               q["metadata"]["difficulty_tier"])
        strat[key].append(q)

    rng = random.Random(42)
    total_written = len(all_written_queries)
    sample: list = []
    for key, group in strat.items():
        n = max(1, round(len(group) / total_written * sample_size))
        sample.extend(rng.sample(group, min(n, len(group))))

    rng.shuffle(sample)
    sample = sample[:sample_size]
    sample_qids = {q["_id"] for q in sample}

    with open(out_dir / "queries_sample.jsonl", "w") as f:
        for q in sample:
            f.write(json.dumps(q) + "\n")

    with open(out_dir / "qrels_sample.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        with open(out_dir / "qrels.tsv") as rf:
            next(rf)
            for line in rf:
                if line.split("\t")[0] in sample_qids:
                    f.write(line)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  corpus.jsonl          : {len(problems):,} documents")
    print(f"  queries.jsonl         : {kept_queries:,} queries")
    print(f"  qrels.tsv             : {total_pairs:,} qrel pairs")
    print(f"  queries_sample.jsonl  : {len(sample)} (stratified, seed=42)")
    print(f"  All outputs in        : {out_dir.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build BEIR-style math reasoning-analogue retrieval dataset"
    )
    ap.add_argument("--input",       default="final_dataset.json",
                    help="Path to final_dataset.json")
    ap.add_argument("--out_dir",     default="dataset",
                    help="Output directory for BEIR files")
    ap.add_argument("--ckpt_dir",    default="checkpoints",
                    help="Checkpoint directory (pipeline is fully resumable)")
    ap.add_argument("--sample_size", type=int, default=SAMPLE_SIZE,
                    help="Size of stratified reporting sample (default: 500)")
    args = ap.parse_args()

    out_dir  = Path(args.out_dir)
    ckpt_dir = Path(args.ckpt_dir)
    out_dir.mkdir(parents=True,  exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading problems from {args.input}...")
    problems = load_problems(args.input)
    print(f"  {len(problems):,} problems loaded\n")

    fps = phase1_fingerprint(problems, ckpt_dir)
    print()

    clusters = phase2_cluster(fps, ckpt_dir)
    print()

    queries, pid_to_query_id = phase3_build_queries(
        problems, fps, clusters, ckpt_dir
    )
    print()

    base_qrels = phase4_qrels(queries, clusters, ckpt_dir)
    print()

    phase5_write(
        problems, fps, queries, base_qrels,
        out_dir, sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()
