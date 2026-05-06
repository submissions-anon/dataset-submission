"""
build_retrieval_dataset.py
--------------------------
Converts tweets_classified_final.jsonl (all tweets) into a BEIR-style retrieval
dataset using only the implicit-labeled tweets.

Pipeline:
  Phase 1 — Describe  : extract implicit meaning of each tweet (batched)
  Phase 2 — Collapse  :
      2a. Assign each description a short theme label (batched, ~80 at a time)
      2b. Collapse near-duplicates into canonical themes (two passes, chunks of 150)
      2c. Re-map every tweet to its canonical theme
  Phase 3 — Query     : for each canonical theme, write a retrieval query + grade relevance
  Phase 4 — Output    : BEIR-format corpus/queries/qrels + human-readable summary
  Phase 5 — Coverage  :
      5a. Embed all implicit_meaning descriptions via OpenAI embeddings
      5b. For each query, retrieve top-50 description-space neighbors
      5c. Strip already-judged IDs
      5d. GPT-5 judges unjudged candidates (relevant=2/1, not=0)
      5e. Append positives to qrels

Requirements:
    pip install openai tqdm numpy

Usage:
    export OPENAI_API_KEY="..."
    python build_retrieval_dataset.py
"""

import json, os, time, math
import numpy as np
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "tweets_classified_final.jsonl"
OUTPUT_DIR  = Path("dataset")
CKPT_DIR    = Path("checkpoints")

MODEL           = "gpt-5"
EMBED_MODEL     = "text-embedding-3-small"

DESCRIBE_BATCH  = 15
LABEL_BATCH     = 80
COLLAPSE_CHUNK  = 150
MIN_GROUP_SIZE  = 3
COVERAGE_TOP_K  = 50   # description-space neighbors per query to judge
JUDGE_BATCH     = 20   # candidates per GPT-5 judgment call

OUTPUT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def gpt(messages, max_tokens=20000):
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=1.0,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return parsed
        except Exception as e:
            print(f"  [gpt] error attempt {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return None

# ── Phase 1: Describe ─────────────────────────────────────────────────────────

DESCRIBE_SYSTEM = """You are an expert in political discourse analysis.
For each tweet, write a SHORT (1-2 sentence) description of its IMPLICIT meaning —
the underlying stance, implication, irony, or subtext that requires inference.
Do NOT describe the literal surface text. Focus on: what is the author really
expressing about governments, military actions, or geopolitical actors?

Respond ONLY with a JSON array in the same order as input:
[{"id": "<tweet_id>", "implicit_meaning": "<description>"}, ...]"""

def phase1_describe(tweets):
    ckpt = CKPT_DIR / "descriptions.json"
    if ckpt.exists():
        print("[Phase 1] Loading descriptions from checkpoint...")
        return json.loads(ckpt.read_text())

    # Resume-safe: load partial results from append-mode JSONL if it exists
    partial_ckpt = CKPT_DIR / "descriptions_partial.jsonl"
    descriptions = {}
    if partial_ckpt.exists():
        with open(partial_ckpt) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    descriptions[entry["id"]] = entry["implicit_meaning"]

    todo = [t for t in tweets if str(t["id"]) not in descriptions]
    print(f"[Phase 1] Describing {len(todo)} tweets ({len(descriptions)} already done)...")

    with open(partial_ckpt, "a") as pf:
        batches = [todo[i:i+DESCRIBE_BATCH] for i in range(0, len(todo), DESCRIBE_BATCH)]
        for batch in tqdm(batches, desc="Describing"):
            block = "\n\n".join(f'Tweet ID {t["id"]}:\n{t["text"]}' for t in batch)
            result = gpt([
                {"role": "system", "content": DESCRIBE_SYSTEM},
                {"role": "user",   "content": block},
            ])
            if result:
                for r, t in zip(result, batch):
                    if isinstance(r, dict):
                        tid  = str(r.get("id", t["id"]))
                        desc = r.get("implicit_meaning", "")
                    else:
                        tid  = str(t["id"])
                        desc = ""
                    descriptions[tid] = desc
                    pf.write(json.dumps({"id": tid, "implicit_meaning": desc}) + "\n")
            else:
                for t in batch:
                    descriptions[str(t["id"])] = ""
                    pf.write(json.dumps({"id": str(t["id"]), "implicit_meaning": ""}) + "\n")
            pf.flush()
            time.sleep(0.3)

    ckpt.write_text(json.dumps(descriptions, indent=2))
    partial_ckpt.unlink(missing_ok=True)
    print(f"  Saved {len(descriptions)} descriptions → {ckpt}")
    return descriptions

# ── Phase 2a: Label each description with a short theme ───────────────────────

LABEL_SYSTEM = """You are grouping political tweets by their underlying implied stance.

For each item (tweet ID + implicit meaning), assign a SHORT theme label (4-8 words)
capturing the core implied stance. Tweets with the same underlying implication
should get the EXACT SAME label string — this is how we group them.

Be specific enough to distinguish different stances, but general enough that
multiple tweets share the same label.

Good label examples:
  "implicit US-Israel military coordination"
  "skepticism toward Western media narratives"
  "civilian casualties minimized by official sources"
  "performative outrage masking political inaction"

Respond ONLY with a JSON array in the same order as input:
[{"id": "<tweet_id>", "theme": "<short label>"}, ...]"""

def phase2a_label(tweets, descriptions):
    ckpt = CKPT_DIR / "raw_labels.json"
    if ckpt.exists():
        print("[Phase 2a] Loading raw labels from checkpoint...")
        return json.loads(ckpt.read_text())

    items = [{"id": str(t["id"]), "implicit_meaning": descriptions.get(str(t["id"]), "")}
             for t in tweets]

    partial_ckpt = CKPT_DIR / "raw_labels_partial.json"
    raw_labels = json.loads(partial_ckpt.read_text()) if partial_ckpt.exists() else {}

    todo = [x for x in items if x["id"] not in raw_labels]
    print(f"[Phase 2a] Labeling {len(todo)} descriptions ({len(raw_labels)} already done)...")

    batches = [todo[i:i+LABEL_BATCH] for i in range(0, len(todo), LABEL_BATCH)]
    for batch in tqdm(batches, desc="Labeling"):
        block = "\n\n".join(f'ID {x["id"]}:\n{x["implicit_meaning"]}' for x in batch)
        result = gpt([
            {"role": "system", "content": LABEL_SYSTEM},
            {"role": "user",   "content": block},
        ])
        if result:
            for r, x in zip(result, batch):
                if isinstance(r, dict):
                    raw_labels[str(r.get("id", x["id"]))] = r.get("theme", "uncategorized")
                else:
                    raw_labels[str(x["id"])] = "uncategorized"
        else:
            for x in batch:
                raw_labels[x["id"]] = "uncategorized"
        partial_ckpt.write_text(json.dumps(raw_labels))
        time.sleep(0.3)

    ckpt.write_text(json.dumps(raw_labels, indent=2))
    partial_ckpt.unlink(missing_ok=True)
    return raw_labels

# ── Phase 2b: Collapse near-duplicate theme labels ────────────────────────────

COLLAPSE_SYSTEM = """You are cleaning up a list of theme labels from political tweet analysis.
Many labels express the same underlying idea with slightly different wording.

Collapse near-duplicates into a single canonical label. Keep genuinely distinct stances separate.

Return ONLY a JSON object mapping every original label to its canonical form:
{"original label": "canonical label", ...}

Rules:
- Every input label must appear as a key
- Pick the clearest, most general phrasing as the canonical
- Many originals can map to the same canonical"""

def run_collapse_pass(labels):
    chunks = [labels[i:i+COLLAPSE_CHUNK] for i in range(0, len(labels), COLLAPSE_CHUNK)]
    merged = {}
    for chunk in tqdm(chunks, desc="  Collapsing", leave=False):
        block = "\n".join(f"- {l}" for l in chunk)
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": COLLAPSE_SYSTEM},
                               {"role": "user",   "content": block}],
                    temperature=1.0,
                    max_completion_tokens=20000,
                    response_format={"type": "json_object"},
                )
                merged.update(json.loads(resp.choices[0].message.content))
                break
            except Exception as e:
                print(f"  [collapse] error attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        time.sleep(0.4)
    return merged

def phase2b_collapse(raw_labels):
    ckpt = CKPT_DIR / "label_collapse.json"
    if ckpt.exists():
        print("[Phase 2b] Loading collapse map from checkpoint...")
        return json.loads(ckpt.read_text())

    unique_raw = sorted(set(raw_labels.values()))
    print(f"[Phase 2b] Collapsing {len(unique_raw)} unique raw labels (pass 1)...")
    pass1 = run_collapse_pass(unique_raw)

    after_pass1 = sorted(set(pass1.values()))
    print(f"  After pass 1: {len(after_pass1)} labels. Running pass 2...")
    pass2 = run_collapse_pass(after_pass1)

    # Chain: raw -> pass1 -> pass2
    collapse_map = {}
    for orig, mid in pass1.items():
        collapse_map[orig] = pass2.get(mid, mid)

    ckpt.write_text(json.dumps(collapse_map, indent=2))
    print(f"  Final canonical labels: {len(set(collapse_map.values()))}")
    return collapse_map

def phase2_collapse(tweets, descriptions):
    ckpt = CKPT_DIR / "theme_groups.json"
    if ckpt.exists():
        print("[Phase 2] Loading theme groups from checkpoint...")
        return json.loads(ckpt.read_text())

    raw_labels  = phase2a_label(tweets, descriptions)
    collapse_map = phase2b_collapse(raw_labels)

    # Map each tweet to its canonical theme
    theme_groups = defaultdict(list)
    for tid, raw_label in raw_labels.items():
        canonical = collapse_map.get(raw_label, raw_label)
        theme_groups[canonical].append(tid)

    theme_groups = {t: tids for t, tids in theme_groups.items()
                    if len(tids) >= MIN_GROUP_SIZE}

    # Filter junk themes
    JUNK_PATTERNS = [
        "insufficient information", "unable to determine", "unspecified",
        "uncategorized", "no clear", "not enough", "unclear",
        "general comment", "no implicit", "no stance", "neutral",
        "miscellaneous", "other", "unknown", "n/a",
    ]
    def is_junk(theme):
        t = theme.lower()
        return any(p in t for p in JUNK_PATTERNS)

    before = len(theme_groups)
    theme_groups = {t: tids for t, tids in theme_groups.items() if not is_junk(t)}
    print(f"  Removed {before - len(theme_groups)} junk themes, {len(theme_groups)} remain")

    print(f"  {len(theme_groups)} themes kept (>={MIN_GROUP_SIZE} tweets)")
    for theme, tids in sorted(theme_groups.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    [{len(tids):3d}] {theme}")
    if len(theme_groups) > 10:
        print("    ...")

    ckpt.write_text(json.dumps(theme_groups, indent=2))
    return dict(theme_groups)

# ── Phase 3: Generate query + relevance per theme ─────────────────────────────

QUERY_SYSTEM = """You are building a hard information retrieval benchmark for political tweet analysis.

You will receive a group of tweets sharing a similar underlying implied stance,
along with each tweet's implicit meaning.

Tasks:
1. Write one RETRIEVAL QUERY a researcher might use to find these tweets.
   - Natural phrasing: "Find tweets where users are..."
   - Must NOT use words that appear verbatim in the tweets
   - Captures the ABSTRACT STANCE or IMPLICATION, not surface content
   - Discriminative: specific enough to exclude unrelated tweets

2. Grade each tweet's relevance to your query:
   - 2 = central example, clearly matches the implied stance
   - 1 = tangentially relevant, partially matches

Respond ONLY with JSON:
{
  "query": "<retrieval query>",
  "relevance": [{"id": "<tweet_id>", "score": 2}, ...]
}"""

def phase3_queries(theme_groups, tweet_by_id, descriptions):
    ckpt = CKPT_DIR / "queries.json"
    if ckpt.exists():
        print("[Phase 3] Loading queries from checkpoint...")
        return json.loads(ckpt.read_text())

    print(f"[Phase 3] Generating queries for {len(theme_groups)} themes...")
    results = []
    for i, (theme, tids) in enumerate(tqdm(theme_groups.items(), desc="Queries")):
        lines = []
        for tid in tids:
            t = tweet_by_id.get(tid)
            if t:
                lines.append(
                    f'Tweet ID {tid}:\n'
                    f'Text: {t["text"]}\n'
                    f'Implicit meaning: {descriptions.get(tid, "")}'
                )
        block = f"Theme: {theme}\n\n" + "\n\n".join(lines)

        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": QUERY_SYSTEM},
                               {"role": "user",   "content": block}],
                    temperature=1.0,
                    max_completion_tokens=20000,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(resp.choices[0].message.content)
                results.append({
                    "theme_id": i,
                    "theme":    theme,
                    "query":    parsed.get("query", ""),
                    "relevance": parsed.get("relevance", []),
                })
                break
            except Exception as e:
                print(f"  [query] error attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        else:
            results.append({"theme_id": i, "theme": theme, "query": "", "relevance": []})
        time.sleep(0.4)

    ckpt.write_text(json.dumps(results, indent=2))
    return results

# ── Phase 5: Coverage expansion via description embeddings ────────────────────

JUDGE_SYSTEM = """You are judging relevance for an information retrieval benchmark on political tweets.

You will receive:
- A RETRIEVAL QUERY (abstract, no verbatim tweet words)
- A set of GOLD tweets confirmed relevant to the query
- A set of CANDIDATE tweets that are unjudged

For each candidate, decide if it is relevant to the query based on its IMPLICIT MEANING,
not its surface text.

Relevance grades:
  2 = clearly relevant: strongly matches the implied stance the query is looking for
  1 = marginally relevant: partially matches, tangentially related
  0 = not relevant: different topic or stance

Respond ONLY with a JSON array:
[{"id": "<tweet_id>", "score": <0|1|2>, "reason": "<one short phrase>"}, ...]"""

def embed_texts(texts, batch_size=500):
    """Embed a list of texts using OpenAI embeddings. Returns np.array (N, dim)."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        for attempt in range(4):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                all_embeddings.extend([e.embedding for e in resp.data])
                break
            except Exception as e:
                print(f"  [embed] error attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        time.sleep(0.2)
    return np.array(all_embeddings, dtype=np.float32)

def cosine_sim_matrix(query_vecs, doc_vecs):
    """query_vecs: (Q, D), doc_vecs: (N, D) -> (Q, N)"""
    q = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9)
    d = doc_vecs   / (np.linalg.norm(doc_vecs,   axis=1, keepdims=True) + 1e-9)
    return q @ d.T

def phase5_coverage(query_results, tweet_by_id, descriptions, existing_qrels):
    """
    existing_qrels: dict { qid -> set of already-judged tweet_ids }
    Returns additional_qrels: dict { qid -> list of (tid, score) }
    """
    ckpt = CKPT_DIR / "coverage_qrels.json"
    if ckpt.exists():
        print("[Phase 5] Loading coverage qrels from checkpoint...")
        return json.loads(ckpt.read_text())

    print("[Phase 5] Coverage expansion via description embeddings...")

    # Build ordered list of all implicit tweets with descriptions
    all_tids  = [tid for tid in tweet_by_id.keys()
                 if descriptions.get(tid, "").strip()]
    all_descs = [descriptions[tid] for tid in all_tids]
    tid_to_idx = {tid: i for i, tid in enumerate(all_tids)}

    print(f"  Embedding {len(all_descs)} descriptions...")
    doc_vecs = embed_texts(all_descs)

    # Embed query texts
    valid_queries = [qr for qr in query_results if qr.get("query")]
    query_texts   = [qr["query"] for qr in valid_queries]
    print(f"  Embedding {len(query_texts)} queries...")
    query_vecs = embed_texts(query_texts)

    # Cosine sim: (Q, N)
    sim = cosine_sim_matrix(query_vecs, doc_vecs)

    additional_qrels = {}
    partial_ckpt = CKPT_DIR / "coverage_qrels_partial.json"
    if partial_ckpt.exists():
        additional_qrels = json.loads(partial_ckpt.read_text())

    for qi, qr in enumerate(tqdm(valid_queries, desc="Judging coverage")):
        qid = f"q{qr['theme_id']:04d}"
        if qid in additional_qrels and additional_qrels[qid] != "FAILED":
            continue

        already_judged = existing_qrels.get(qid, set())

        # Top-K by description-space similarity, excluding already judged
        scores = sim[qi]
        top_idxs = np.argsort(-scores)
        candidates = []
        for idx in top_idxs:
            tid = all_tids[idx]
            if tid not in already_judged:
                candidates.append(tid)
            if len(candidates) >= COVERAGE_TOP_K:
                break

        if not candidates:
            additional_qrels[qid] = []
            continue

        # Gold tweets for context
        gold_tids = [str(r["id"]) for r in qr.get("relevance", []) if r.get("score", 0) >= 2]
        gold_lines = []
        for tid in gold_tids[:5]:  # cap at 5 gold examples for prompt length
            t = tweet_by_id.get(tid)
            if t:
                gold_lines.append(
                    f'  ID {tid}: {t["text"]}\n'
                    f'  Implicit meaning: {descriptions.get(tid, "")}'
                )

        # Judge in batches
        new_judgments = []
        cand_batches = [candidates[i:i+JUDGE_BATCH]
                        for i in range(0, len(candidates), JUDGE_BATCH)]
        for cbatch in cand_batches:
            cand_lines = []
            for tid in cbatch:
                t = tweet_by_id.get(tid)
                if t:
                    cand_lines.append(
                        f'  ID {tid}: {t["text"]}\n'
                        f'  Implicit meaning: {descriptions.get(tid, "")}'
                    )

            user_block = (
                f"QUERY: {qr['query']}\n\n"
                f"GOLD RELEVANT TWEETS (for reference):\n"
                + "\n\n".join(gold_lines) +
                f"\n\nCANDIDATE TWEETS TO JUDGE:\n"
                + "\n\n".join(cand_lines)
            )

            result = gpt([
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_block},
            ])
            if result:
                for r in result:
                    if not isinstance(r, dict):
                        continue
                    score = int(r.get("score", 0))
                    if score > 0:
                        new_judgments.append({"id": str(r["id"]), "score": score})
            time.sleep(0.3)

        # If every batch failed (no judgments at all and candidates existed),
        # save a FAILED sentinel so resume will retry this query
        if not new_judgments and candidates:
            additional_qrels[qid] = "FAILED"
        else:
            additional_qrels[qid] = new_judgments
        partial_ckpt.write_text(json.dumps(additional_qrels))
        time.sleep(0.3)

    ckpt.write_text(json.dumps(additional_qrels, indent=2))
    partial_ckpt.unlink(missing_ok=True)

    total_new = sum(len(v) for v in additional_qrels.values())
    print(f"  Found {total_new} additional relevant tweets across {len(additional_qrels)} queries")
    return additional_qrels

# ── Phase 4: Write BEIR output ────────────────────────────────────────────────

def phase4_write(tweets, descriptions, query_results, tweet_by_id,
                 theme_groups, additional_qrels=None, all_tweets=None):
    print("[Phase 4] Writing output files...")

    additional_qrels = additional_qrels or {}

    # Build theme_id -> cluster tids
    theme_id_to_tids = {}
    for qr in query_results:
        theme = qr.get("theme", "")
        if theme in theme_groups:
            theme_id_to_tids[qr["theme_id"]] = set(theme_groups[theme])

    def tweet_to_doc(t):
        tid = str(t["id"])
        return json.dumps({
            "_id":  tid,
            "text": t["text"],
            "metadata": {
                "author_id":        t.get("author_id", ""),
                "created_at":       t.get("created_at", ""),
                "implicit_meaning": descriptions.get(tid, ""),
                "label":            t.get("_label", ""),
                "lang":             t.get("lang", "en"),
                "public_metrics":   t.get("public_metrics", {}),
            }
        })

    # corpus_implicit.jsonl — only the 7,918 implicit tweets
    with open(OUTPUT_DIR / "corpus_implicit.jsonl", "w") as f:
        for t in tweets:
            f.write(tweet_to_doc(t) + "\n")

    # corpus_full.jsonl — all 72k tweets (explicit + news_repost + implicit)
    if all_tweets:
        with open(OUTPUT_DIR / "corpus_full.jsonl", "w") as f:
            for t in all_tweets:
                f.write(tweet_to_doc(t) + "\n")

    kept = 0
    total_qrel_pairs = 0

    with open(OUTPUT_DIR / "queries.jsonl", "w") as qf, \
         open(OUTPUT_DIR / "qrels.tsv",     "w") as rf, \
         open(OUTPUT_DIR / "summary.jsonl", "w") as sf:

        rf.write("query-id\tcorpus-id\tscore\n")

        for qr in query_results:
            if not qr.get("query"):
                continue
            qid = f"q{qr['theme_id']:04d}"

            qf.write(json.dumps({
                "_id":  qid,
                "text": qr["query"],
                "metadata": {"theme": qr.get("theme", "")}
            }) + "\n")

            scored_ids = {}  # tid -> score (highest wins on conflict)

            # Cluster members GPT omitted default to score=1
            cluster_tids = theme_id_to_tids.get(qr["theme_id"], set())
            for tid in cluster_tids:
                if tid in tweet_by_id:
                    scored_ids[tid] = max(scored_ids.get(tid, 0), 1)

            # Phase 3 explicit relevance grades
            for rel in qr.get("relevance", []):
                tid   = str(rel["id"])
                score = int(rel.get("score", 1))
                if score > 0 and tid in tweet_by_id:
                    scored_ids[tid] = max(scored_ids.get(tid, 0), score)

            # Phase 5 coverage expansion
            phase5_rels = additional_qrels.get(qid)
            for rel in (phase5_rels if isinstance(phase5_rels, list) else []):
                tid   = str(rel["id"])
                score = int(rel.get("score", 0))
                if score > 0 and tid in tweet_by_id:
                    scored_ids[tid] = max(scored_ids.get(tid, 0), score)

            relevant_tweets = []
            for tid, score in scored_ids.items():
                rf.write(f"{qid}\t{tid}\t{score}\n")
                t = tweet_by_id[tid]
                relevant_tweets.append({
                    "id":               tid,
                    "text":             t["text"],
                    "implicit_meaning": descriptions.get(tid, ""),
                    "relevance_score":  score,
                })
                total_qrel_pairs += 1

            relevant_tweets.sort(key=lambda x: -x["relevance_score"])
            sf.write(json.dumps({
                "query_id":        qid,
                "theme":           qr.get("theme", ""),
                "query":           qr["query"],
                "n_relevant":      len(relevant_tweets),
                "relevant_tweets": relevant_tweets,
            }) + "\n")
            kept += 1

    n_full = len(all_tweets) if all_tweets else 0
    print(f"\nDone.")
    print(f"  Corpus (implicit) : {len(tweets)} tweets   → {OUTPUT_DIR}/corpus_implicit.jsonl")
    print(f"  Corpus (full)     : {n_full} tweets        → {OUTPUT_DIR}/corpus_full.jsonl")
    print(f"  Queries           : {kept}                 → {OUTPUT_DIR}/queries.jsonl")
    print(f"  Qrels             : {total_qrel_pairs} pairs → {OUTPUT_DIR}/qrels.tsv")
    print(f"  Summary           :                        → {OUTPUT_DIR}/summary.jsonl")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load only implicit tweets
    all_tweets = load_jsonl(INPUT_FILE)
    tweets = [t for t in all_tweets if t.get("_label") == "implicit"]
    print(f"Loaded {len(all_tweets):,} total tweets, {len(tweets):,} implicit")

    tweet_by_id = {str(t["id"]): t for t in tweets}

    descriptions  = phase1_describe(tweets)
    theme_groups  = phase2_collapse(tweets, descriptions)
    query_results = phase3_queries(theme_groups, tweet_by_id, descriptions)

    # Build existing qrels before phase 5 so we know what's already judged
    existing_qrels = defaultdict(set)
    for qr in query_results:
        qid = f"q{qr['theme_id']:04d}"
        for rel in qr.get("relevance", []):
            existing_qrels[qid].add(str(rel["id"]))
        for tid in theme_groups.get(qr.get("theme", ""), []):
            existing_qrels[qid].add(str(tid))

    additional_qrels = phase5_coverage(
        query_results, tweet_by_id, descriptions, existing_qrels
    )

    phase4_write(tweets, descriptions, query_results, tweet_by_id,
                 theme_groups, additional_qrels, all_tweets=all_tweets)

if __name__ == "__main__":
    main()
