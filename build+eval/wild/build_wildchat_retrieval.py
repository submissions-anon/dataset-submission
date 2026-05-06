"""
build_wildchat_retrieval.py
---------------------------
Builds a BEIR-style retrieval benchmark from WildChat-2025-English
where queries describe LLM mistake types and documents are conversations
exhibiting those mistakes.

Pipeline:
  Phase 1 — Sweep    : stream all gz chunks, call nano on each batch,
                       write sweep.json (mistake_type + description or null)
  Phase 2 — Collapse :
      2a. Assign each flagged conv a short canonical mistake label
      2b. Collapse near-duplicates into canonical types (two passes)
      2c. Re-map every conv to its canonical type
  Phase 3 — Query    : for each canonical type, write a retrieval query
                       + grade relevance of cluster members
  Phase 4 — Output   : BEIR-format corpus/queries/qrels + summary
  Phase 5 — Coverage : embed mistake_descriptions, expand qrels via
                       embedding search + LLM judge

NOTE on IDs: conversation_hash is NOT unique (same content = same hash).
We assign a sequential _row_id per conversation based on chunk file order.
Row IDs are stable as long as chunk files don't change between runs.

Usage:
    export OPENAI_API_KEY="..."
    python build_wildchat_retrieval.py
"""

import json, os, time, gzip, glob
import numpy as np
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR   = "FINAL-DATA/wildchat_2025_chunks"   # directory of *.jsonl.gz files
OUTPUT_DIR  = Path("dataset")
CKPT_DIR    = Path("checkpoints")

MODEL           = "gpt-5.4-nano"        # cheap sweep model (Phase 1 + 2)
JUDGE_MODEL     = "gpt-5"        # stronger for Phase 3 + 5
EMBED_MODEL     = "text-embedding-3-small"

SWEEP_BATCH     = 4     # conversations per Phase 1 API call (they're long)
LABEL_BATCH     = 80    # descriptions per Phase 2a call
COLLAPSE_CHUNK  = 150   # labels per collapse pass
MIN_GROUP_SIZE  = 5     # minimum convs per mistake type to keep
COVERAGE_TOP_K  = 50    # description-space neighbors per query to judge
JUDGE_BATCH     = 15    # candidates per judgment call
MAX_CONV_CHARS  = 10000  # truncate very long conversations to this many chars

OUTPUT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def iter_convs():
    """
    Stream conversations one at a time from all gz chunks.
    Assigns a guaranteed-unique sequential _row_id to each conversation.
    Stable across runs as long as chunk files don't change.
    Never loads the full dataset into RAM.
    """
    chunk_files = sorted(glob.glob(f"{INPUT_DIR}/*.jsonl.gz"))
    if not chunk_files:
        raise FileNotFoundError(f"No .jsonl.gz files found in {INPUT_DIR}")
    row_idx = 0
    for chunk_file in chunk_files:
        with gzip.open(chunk_file, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    c["_row_id"] = str(row_idx)
                    row_idx += 1
                    yield c

def get_conv_id(c):
    """Primary unique ID for a conversation."""
    return c["_row_id"]

def format_conversation(c):
    """
    Extract only role+content from the raw WildChat conversation field.
    Strips all metadata (IP, country, headers, etc.) to avoid biasing the LLM.
    Truncates to MAX_CONV_CHARS.
    """
    turns = c.get("conversation", [])
    lines = []
    for turn in turns:
        role    = (turn.get("role") or "unknown").capitalize()
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"[{role}]: {content}")
    full = "\n\n".join(lines)
    if len(full) > MAX_CONV_CHARS:
        full = full[:MAX_CONV_CHARS] + "\n[... truncated ...]"
    return full

def gpt(messages, model=None, max_tokens=20000):
    model = model or MODEL
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw    = resp.choices[0].message.content
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

def flush_batch(batch, results):
    block = "\n\n---\n\n".join(
        f'Conversation ID {get_conv_id(c)}:\n{format_conversation(c)}'
        for c in batch
    )
    result = gpt([
        {"role": "system", "content": SWEEP_SYSTEM},
        {"role": "user",   "content": block},
    ])
    if result and isinstance(result, list):
        for r, c in zip(result, batch):
            cid = get_conv_id(c)
            if isinstance(r, dict):
                results[str(r.get("id", cid))] = {
                    "mistake_type":        r.get("mistake_type"),
                    "mistake_description": r.get("mistake_description"),
                    "conversation_hash":   c.get("conversation_hash", ""),
                }
            else:
                # LLM returned a string or malformed item — treat as no mistake
                results[cid] = {
                    "mistake_type":        None,
                    "mistake_description": None,
                    "conversation_hash":   c.get("conversation_hash", ""),
                }
    else:
        for c in batch:
            results[get_conv_id(c)] = {
                "mistake_type":        None,
                "mistake_description": None,
                "conversation_hash":   c.get("conversation_hash", ""),
            }
    time.sleep(0.3)


# ── Phase 1: Streaming sweep for mistakes ─────────────────────────────────────

SWEEP_SYSTEM = """You are an expert at identifying LLM failure modes in human-AI conversations.

For each conversation, determine whether the AI assistant makes a notable, specific mistake.

If a clear mistake is present, return:
  - mistake_type: a short label (4-8 words) for the failure category
  - mistake_description: 1-2 sentences describing the specific failure in this conversation

If no clear mistake is present, return null for both fields.

Focus on behavioral failures such as:
  - Confident hallucination (states false facts without hedging)
  - Sycophantic capitulation (abandons a correct position under user pushback)
  - False precision (fabricates specific numbers, dates, citations)
  - Instruction drift (forgets earlier constraints mid-conversation)
  - Over-refusal (declines a clearly benign request citing safety)
  - Misattributed citation (invents a paper, quote, or source)
  - Calculation or unit error (arithmetic or conversion mistake)
  - Context drop (forgets something stated earlier in the conversation)
  - Unwarranted flattery (praises an obviously flawed idea)
  - Persona bleed (breaks or over-applies a roleplay character)

Do NOT invent mistakes. Only flag failures clearly visible in the text.
Be conservative: if unsure, return null.

Respond ONLY with a JSON array in the same order as input:
[{"id": "<conv_id>", "mistake_type": "<label or null>", "mistake_description": "<1-2 sentences or null>"}, ...]"""

def phase1_sweep():
    ckpt = CKPT_DIR / "sweep.json"
    if ckpt.exists():
        print("[Phase 1] Loading sweep results from checkpoint...")
        return json.loads(ckpt.read_text())

    partial_ckpt = CKPT_DIR / "sweep_partial.json"
    results = json.loads(partial_ckpt.read_text()) if partial_ckpt.exists() else {}
    already_done = len(results)
    print(f"[Phase 1] Streaming sweep ({already_done:,} already done)...")

    chunk_files = sorted(glob.glob(f"{INPUT_DIR}/*.jsonl.gz"))
    print(f"  Found {len(chunk_files)} chunk files: {[Path(f).name for f in chunk_files]}")

    batch = []
    processed_since_save = 0
    save_every = 500

    with tqdm(desc="Sweeping", unit="conv") as pbar:
        for c in iter_convs():
            cid = get_conv_id(c)
            if cid in results:
                pbar.update(1)
                continue

            batch.append(c)

            if len(batch) >= SWEEP_BATCH:
                flush_batch(batch, results)
                processed_since_save += len(batch)
                pbar.update(len(batch))
                batch = []

                if processed_since_save >= save_every:
                    partial_ckpt.write_text(json.dumps(results))
                    processed_since_save = 0

        # flush remaining partial batch
        if batch:
            flush_batch(batch, results)
            pbar.update(len(batch))

    partial_ckpt.write_text(json.dumps(results))

    flagged = sum(1 for v in results.values() if v.get("mistake_type"))
    print(f"  Flagged {flagged:,}/{len(results):,} conversations ({100*flagged/max(len(results),1):.1f}%)")

    ckpt.write_text(json.dumps(results, indent=2))
    partial_ckpt.unlink(missing_ok=True)
    return results

# ── Phase 2a: Label each flagged description ──────────────────────────────────

LABEL_SYSTEM = """You are grouping LLM failure modes by their underlying mistake category.

For each item (conversation ID + mistake description), assign a SHORT canonical
theme label (4-8 words) capturing the core failure type. Conversations with the
same kind of mistake should get the EXACT SAME label string.

Be specific enough to distinguish different failure modes, but general enough
that multiple conversations share the same label.

Good label examples:
  "confident hallucination of factual claims"
  "sycophantic capitulation under user pressure"
  "fabricated citation or source"
  "arithmetic or unit conversion error"
  "over-refusal of benign request"
  "context window drop mid-conversation"

Respond ONLY with a JSON array in the same order as input:
[{"id": "<conv_id>", "theme": "<short label>"}, ...]"""

def phase2a_label(flagged_convs, sweep_results):
    ckpt = CKPT_DIR / "raw_labels.json"
    if ckpt.exists():
        print("[Phase 2a] Loading raw labels from checkpoint...")
        return json.loads(ckpt.read_text())

    items = [
        {"id": c["id"], "mistake_description": sweep_results[c["id"]]["mistake_description"]}
        for c in flagged_convs
    ]

    partial_ckpt = CKPT_DIR / "raw_labels_partial.json"
    raw_labels = json.loads(partial_ckpt.read_text()) if partial_ckpt.exists() else {}

    todo = [x for x in items if x["id"] not in raw_labels]
    print(f"[Phase 2a] Labeling {len(todo):,} descriptions ({len(raw_labels):,} already done)...")

    batches = [todo[i:i+LABEL_BATCH] for i in range(0, len(todo), LABEL_BATCH)]
    for batch in tqdm(batches, desc="Labeling"):
        block = "\n\n".join(
            f'ID {x["id"]}:\n{x["mistake_description"]}'
            for x in batch
        )
        result = gpt([
            {"role": "system", "content": LABEL_SYSTEM},
            {"role": "user",   "content": block},
        ])
        if result:
            for r, x in zip(result, batch):
                if isinstance(r, dict):
                    raw_labels[str(r.get("id", x["id"]))] = r.get("theme", "uncategorized")
                else:
                    raw_labels[x["id"]] = "uncategorized"
        else:
            for x in batch:
                raw_labels[x["id"]] = "uncategorized"
        partial_ckpt.write_text(json.dumps(raw_labels))
        time.sleep(0.3)

    ckpt.write_text(json.dumps(raw_labels, indent=2))
    partial_ckpt.unlink(missing_ok=True)
    return raw_labels

# ── Phase 2b: Collapse near-duplicate theme labels ────────────────────────────

COLLAPSE_SYSTEM = """You are cleaning up a list of LLM failure mode labels.
Many labels express the same underlying mistake type with slightly different wording.

Collapse near-duplicates into a single canonical label. Keep genuinely distinct
failure types separate.

Return ONLY a JSON object mapping every original label to its canonical form:
{"original label": "canonical label", ...}

Rules:
- Every input label must appear as a key
- Pick the clearest, most descriptive phrasing as the canonical
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
                    messages=[
                        {"role": "system", "content": COLLAPSE_SYSTEM},
                        {"role": "user",   "content": block},
                    ],
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

    collapse_map = {}
    for orig, mid in pass1.items():
        collapse_map[orig] = pass2.get(mid, mid)

    ckpt.write_text(json.dumps(collapse_map, indent=2))
    print(f"  Final canonical types: {len(set(collapse_map.values()))}")
    return collapse_map

def phase2_collapse(flagged_convs, sweep_results):
    ckpt = CKPT_DIR / "type_groups.json"
    if ckpt.exists():
        print("[Phase 2] Loading type groups from checkpoint...")
        return json.loads(ckpt.read_text())

    raw_labels   = phase2a_label(flagged_convs, sweep_results)
    collapse_map = phase2b_collapse(raw_labels)

    type_groups = defaultdict(list)
    for cid, raw_label in raw_labels.items():
        canonical = collapse_map.get(raw_label, raw_label)
        type_groups[canonical].append(cid)

    type_groups = {t: cids for t, cids in type_groups.items()
                   if len(cids) >= MIN_GROUP_SIZE}

    print(f"  {len(type_groups)} mistake types kept (>={MIN_GROUP_SIZE} convs)")
    for mtype, cids in sorted(type_groups.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"    [{len(cids):4d}] {mtype}")
    if len(type_groups) > 15:
        print("    ...")

    ckpt.write_text(json.dumps(type_groups, indent=2))
    return dict(type_groups)

# ── Phase 3: Generate query + relevance per mistake type ──────────────────────

QUERY_SYSTEM = """You are building a hard information retrieval benchmark for LLM failure mode analysis.

You will receive a group of AI-human conversations that share a common LLM mistake type,
along with a brief description of the mistake in each conversation.

Tasks:
1. Write one RETRIEVAL QUERY a researcher might use to find these conversations.
   - Natural phrasing: "Find conversations where the AI..."
   - Must NOT use words from the mistake descriptions verbatim
   - Captures the ABSTRACT FAILURE PATTERN, not surface content
   - Discriminative: specific enough to exclude unrelated failure modes

2. Grade each conversation's relevance to your query:
   - 2 = central example, clearly exemplifies the failure pattern
   - 1 = tangentially relevant, partially matches

Respond ONLY with JSON:
{
  "query": "<retrieval query>",
  "relevance": [{"id": "<conv_id>", "score": 2}, ...]
}"""

def phase3_queries(type_groups, conv_by_id, sweep_results):
    ckpt = CKPT_DIR / "queries.json"
    if ckpt.exists():
        print("[Phase 3] Loading queries from checkpoint...")
        return json.loads(ckpt.read_text())

    print(f"[Phase 3] Generating queries for {len(type_groups)} mistake types...")
    results = []
    for i, (mtype, cids) in enumerate(tqdm(type_groups.items(), desc="Queries")):
        lines = []
        for cid in cids[:20]:
            desc = sweep_results.get(cid, {}).get("mistake_description", "")
            if desc:
                lines.append(f'Conv ID {cid}:\nMistake: {desc}')

        block = f"Mistake type: {mtype}\n\n" + "\n\n".join(lines)

        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": QUERY_SYSTEM},
                        {"role": "user",   "content": block},
                    ],
                    temperature=1.0,
                    max_completion_tokens=20000,
                    response_format={"type": "json_object"},
                )
                parsed = json.loads(resp.choices[0].message.content)
                results.append({
                    "type_id":   i,
                    "mtype":     mtype,
                    "query":     parsed.get("query", ""),
                    "relevance": parsed.get("relevance", []),
                })
                break
            except Exception as e:
                print(f"  [query] error attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        else:
            results.append({"type_id": i, "mtype": mtype, "query": "", "relevance": []})
        time.sleep(0.4)

    ckpt.write_text(json.dumps(results, indent=2))
    return results

# ── Phase 5: Coverage expansion via description embeddings ────────────────────

JUDGE_SYSTEM = """You are judging relevance for an information retrieval benchmark on LLM failure modes.

You will receive:
- A RETRIEVAL QUERY describing an LLM failure pattern
- GOLD conversations confirmed to exhibit the failure
- CANDIDATE conversations that are unjudged

For each candidate, decide if it exhibits the failure pattern the query describes.
Base your judgment on the mistake description, not surface topic.

Relevance grades:
  2 = clearly relevant: strongly exemplifies the failure pattern
  1 = marginally relevant: partially matches, similar but weaker failure
  0 = not relevant: different failure type or no mistake

Respond ONLY with a JSON array:
[{"id": "<conv_id>", "score": <0|1|2>, "reason": "<one short phrase>"}, ...]"""

def embed_texts(texts, batch_size=500):
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
    q = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9)
    d = doc_vecs   / (np.linalg.norm(doc_vecs,   axis=1, keepdims=True) + 1e-9)
    return q @ d.T

def phase5_coverage(query_results, sweep_results, existing_qrels):
    ckpt = CKPT_DIR / "coverage_qrels.json"
    if ckpt.exists():
        print("[Phase 5] Loading coverage qrels from checkpoint...")
        return json.loads(ckpt.read_text())

    print("[Phase 5] Coverage expansion via description embeddings...")

    all_cids  = [cid for cid, v in sweep_results.items()
                 if v.get("mistake_description", "")]
    all_descs = [sweep_results[cid]["mistake_description"] for cid in all_cids]

    print(f"  Embedding {len(all_descs):,} mistake descriptions...")
    doc_vecs = embed_texts(all_descs)

    valid_queries = [qr for qr in query_results if qr.get("query")]
    query_texts   = [qr["query"] for qr in valid_queries]
    print(f"  Embedding {len(query_texts)} queries...")
    query_vecs = embed_texts(query_texts)

    sim = cosine_sim_matrix(query_vecs, doc_vecs)

    additional_qrels = {}
    partial_ckpt = CKPT_DIR / "coverage_qrels_partial.json"
    if partial_ckpt.exists():
        additional_qrels = json.loads(partial_ckpt.read_text())

    for qi, qr in enumerate(tqdm(valid_queries, desc="Judging coverage")):
        qid = f"q{qr['type_id']:04d}"
        if qid in additional_qrels:
            continue

        already_judged = existing_qrels.get(qid, set())
        scores   = sim[qi]
        top_idxs = np.argsort(-scores)
        candidates = []
        for idx in top_idxs:
            cid = all_cids[idx]
            if cid not in already_judged:
                candidates.append(cid)
            if len(candidates) >= COVERAGE_TOP_K:
                break

        if not candidates:
            additional_qrels[qid] = []
            continue

        gold_cids  = [str(r["id"]) for r in qr.get("relevance", []) if r.get("score", 0) >= 2]
        gold_lines = []
        for cid in gold_cids[:3]:
            desc = sweep_results.get(cid, {}).get("mistake_description", "")
            if desc:
                gold_lines.append(f'  ID {cid}: {desc}')

        new_judgments = []
        cand_batches  = [candidates[i:i+JUDGE_BATCH]
                         for i in range(0, len(candidates), JUDGE_BATCH)]
        for cbatch in cand_batches:
            cand_lines = []
            for cid in cbatch:
                desc = sweep_results.get(cid, {}).get("mistake_description", "")
                if desc:
                    cand_lines.append(f'  ID {cid}: {desc}')

            user_block = (
                f"QUERY: {qr['query']}\n\n"
                f"GOLD RELEVANT CONVERSATIONS (for reference):\n"
                + "\n".join(gold_lines) +
                f"\n\nCANDIDATE CONVERSATIONS TO JUDGE:\n"
                + "\n".join(cand_lines)
            )

            result = gpt([
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_block},
            ], model=JUDGE_MODEL)

            if result:
                for r in result:
                    if not isinstance(r, dict):
                        continue
                    score = int(r.get("score", 0))
                    if score > 0:
                        new_judgments.append({"id": str(r["id"]), "score": score})
            time.sleep(0.3)

        additional_qrels[qid] = new_judgments
        partial_ckpt.write_text(json.dumps(additional_qrels))
        time.sleep(0.3)

    ckpt.write_text(json.dumps(additional_qrels, indent=2))
    partial_ckpt.unlink(missing_ok=True)
    total_new = sum(len(v) for v in additional_qrels.values())
    print(f"  Found {total_new} additional relevant convs across {len(additional_qrels)} queries")
    return additional_qrels

# ── Phase 4: Write BEIR output ────────────────────────────────────────────────

def phase4_write(flagged_convs, sweep_results, query_results, conv_by_id,
                 type_groups, additional_qrels=None):
    print("[Phase 4] Writing BEIR output files...")
    additional_qrels = additional_qrels or {}

    type_id_to_cids = {}
    for qr in query_results:
        mtype = qr.get("mtype", "")
        if mtype in type_groups:
            type_id_to_cids[qr["type_id"]] = set(type_groups[mtype])

    def conv_to_doc(c):
        cid = c["id"]
        sr  = sweep_results.get(cid, {})
        return json.dumps({
            "_id":  cid,
            "text": format_conversation(c),
            "metadata": {
                "mistake_type":        sr.get("mistake_type"),
                "mistake_description": sr.get("mistake_description"),
                "conversation_hash":   c.get("conversation_hash", ""),
                "turn":                c.get("turn", 1),
            }
        })

    print("  Writing full corpus (streaming all conversations)...")
    with open(OUTPUT_DIR / "corpus.jsonl", "w") as f:
        for c in tqdm(iter_convs(), desc="Writing corpus", unit="conv"):
            c["id"] = get_conv_id(c)
            f.write(conv_to_doc(c) + "\n")

    kept             = 0
    total_qrel_pairs = 0

    with open(OUTPUT_DIR / "queries.jsonl", "w") as qf, \
         open(OUTPUT_DIR / "qrels.tsv",     "w") as rf, \
         open(OUTPUT_DIR / "summary.jsonl", "w") as sf:

        rf.write("query-id\tcorpus-id\tscore\n")

        for qr in query_results:
            if not qr.get("query"):
                continue
            qid = f"q{qr['type_id']:04d}"

            qf.write(json.dumps({
                "_id":  qid,
                "text": qr["query"],
                "metadata": {"mistake_type": qr.get("mtype", "")}
            }) + "\n")

            scored_ids = {}

            for cid in type_id_to_cids.get(qr["type_id"], set()):
                if cid in conv_by_id:
                    scored_ids[cid] = max(scored_ids.get(cid, 0), 1)

            for rel in qr.get("relevance", []):
                cid   = str(rel["id"])
                score = int(rel.get("score", 1))
                if score > 0 and cid in conv_by_id:
                    scored_ids[cid] = max(scored_ids.get(cid, 0), score)

            for rel in additional_qrels.get(qid, []):
                cid   = str(rel["id"])
                score = int(rel.get("score", 0))
                if score > 0 and cid in conv_by_id:
                    scored_ids[cid] = max(scored_ids.get(cid, 0), score)

            relevant_convs = []
            for cid, score in scored_ids.items():
                rf.write(f"{qid}\t{cid}\t{score}\n")
                relevant_convs.append({
                    "id":                  cid,
                    "mistake_description": sweep_results.get(cid, {}).get("mistake_description", ""),
                    "relevance_score":     score,
                })
                total_qrel_pairs += 1

            relevant_convs.sort(key=lambda x: -x["relevance_score"])
            sf.write(json.dumps({
                "query_id":       qid,
                "mistake_type":   qr.get("mtype", ""),
                "query":          qr["query"],
                "n_relevant":     len(relevant_convs),
                "relevant_convs": relevant_convs,
            }) + "\n")
            kept += 1

    print(f"\nDone.")
    print(f"  Corpus  : {len(flagged_convs):,} conversations → {OUTPUT_DIR}/corpus.jsonl")
    print(f"  Queries : {kept}                               → {OUTPUT_DIR}/queries.jsonl")
    print(f"  Qrels   : {total_qrel_pairs} pairs             → {OUTPUT_DIR}/qrels.tsv")
    print(f"  Summary :                                      → {OUTPUT_DIR}/summary.jsonl")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    chunk_files = sorted(glob.glob(f"{INPUT_DIR}/*.jsonl.gz"))
    print(f"Found {len(chunk_files)} chunk files in {INPUT_DIR}:")
    for f in chunk_files:
        print(f"  {Path(f).name}")

    # ── Phase 1: stream all chunks, never load full dataset into RAM ──────────
    sweep_results = phase1_sweep()

    # ── Second pass: collect only flagged convs into memory (small) ───────────
    flagged_ids = {cid for cid, v in sweep_results.items() if v.get("mistake_type")}
    print(f"\nSecond pass: collecting {len(flagged_ids):,} flagged conversations...")

    flagged_convs = []
    for c in tqdm(iter_convs(), desc="Collecting flagged", unit="conv"):
        cid = get_conv_id(c)
        if cid in flagged_ids:
            c["id"] = cid   # normalize id field
            flagged_convs.append(c)

    conv_by_id = {c["id"]: c for c in flagged_convs}
    print(f"  Loaded {len(flagged_convs):,} flagged convs into memory")

    # ── Phases 2-5: operate only on flagged subset ────────────────────────────
    type_groups   = phase2_collapse(flagged_convs, sweep_results)
    query_results = phase3_queries(type_groups, conv_by_id, sweep_results)

    existing_qrels = defaultdict(set)
    for qr in query_results:
        qid = f"q{qr['type_id']:04d}"
        for rel in qr.get("relevance", []):
            existing_qrels[qid].add(str(rel["id"]))
        for cid in type_groups.get(qr.get("mtype", ""), []):
            existing_qrels[qid].add(str(cid))

    additional_qrels = phase5_coverage(query_results, sweep_results, existing_qrels)

    phase4_write(flagged_convs, sweep_results, query_results, conv_by_id,
                 type_groups, additional_qrels)

if __name__ == "__main__":
    main()
