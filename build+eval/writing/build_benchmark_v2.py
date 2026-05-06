"""
build_benchmark_v2.py
=====================
From corpus_track.csv (gold) + distractor_track.csv (distractors) + the
corpus/ directory of .txt files, produce the files the eval scripts expect:

  corpus.jsonl              — every snippet as {"_id": "N", "text": "..."}
  queries.jsonl             — only snippets from authors with >= --min_per_author
                              gold snippets. These become queries.
  qrels.tsv                 — for each query, gold = other snippets by same
                              author (same-post excluded). Format:
                                  query_id \t corpus_id \t score
                              Score is always 1 (binary qrels).
  per_query_excluded_ids.json — {query_id: [query_id, any_same_post_ids]}.
                              With one-snippet-per-post gold, the excluded
                              set is just {query_id}. With multi-snippet-
                              per-post gold (if any), same-post snippets
                              are also excluded at retrieval time.

Usage:
    python build_benchmark_v2.py \
        --gold_track corpus_track.csv \
        --distractor_track distractor_track.csv \
        --corpus_dir corpus/ \
        --out_dir benchmark_v2/ \
        --min_per_author 6
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_corpus_track(path, key_name, required_cols):
    """Read a CSV, return list of dicts. Skip rows with empty required_cols."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        # Normalize header names (strip whitespace)
        fieldnames = [fn.strip() for fn in reader.fieldnames or []]
        reader.fieldnames = fieldnames
        for row in reader:
            clean = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if all(clean.get(col) for col in required_cols):
                rows.append(clean)
    print(f"  {path.name}: loaded {len(rows)} rows ({key_name})")
    return rows


def read_snippet(path: Path) -> str:
    """Read a snippet file, tolerating non-UTF-8 bytes (replaced with ?)."""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold_track",        default="corpus_track.csv",
                   help="Gold corpus track CSV with columns: snippet_id, author_name, post_title, post_url")
    p.add_argument("--distractor_track",  default="distractor_track.csv",
                   help="Distractor corpus track CSV with columns: snippet_id, source, author, title, url")
    p.add_argument("--corpus_dir",        default="corpus/",
                   help="Directory containing N.txt files for every snippet_id")
    p.add_argument("--out_dir",           default="benchmark_v2/",
                   help="Output directory for corpus.jsonl, queries.jsonl, qrels.tsv, per_query_excluded_ids.json")
    p.add_argument("--min_per_author",    type=int, default=6,
                   help="Authors with >= this many gold snippets become query authors. Others contribute to pool only.")
    args = p.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading tracks...")
    gold = load_corpus_track(
        Path(args.gold_track),
        key_name="gold",
        required_cols=["snippet_id", "author_name", "post_title"],
    )
    dist = load_corpus_track(
        Path(args.distractor_track),
        key_name="distractor",
        required_cols=["snippet_id"],
    )

    # ── Author/post metadata keyed by snippet_id ─────────────────────────────
    # The CSVs determine GOLD/QUERY status only. Corpus membership comes from
    # the filesystem — every .txt in corpus_dir is part of the retrieval pool,
    # regardless of whether it's tracked in either CSV. This lets you demote
    # an author (remove them from corpus_track.csv) without losing their
    # snippets from the corpus pool; they just become untracked pool items.
    snippet_author = {}
    snippet_post   = {}
    snippet_url    = {}
    is_gold        = {}

    for row in gold:
        sid = row["snippet_id"]
        snippet_author[sid] = row["author_name"]
        snippet_post[sid]   = row.get("post_title", "")
        snippet_url[sid]    = row.get("post_url", "")
        is_gold[sid]        = True

    for row in dist:
        sid = row["snippet_id"]
        # Don't let distractor CSV clobber a gold entry if IDs overlap (shouldn't, but defensive)
        if sid in snippet_author and is_gold[sid]:
            continue
        snippet_author[sid] = row.get("author", "") or f"distractor_{sid}"
        snippet_post[sid]   = row.get("title", "")
        snippet_url[sid]    = row.get("url", "")
        is_gold[sid]        = False

    # ── Discover all corpus files from disk ─────────────────────────────────
    print("\n[2/5] Scanning corpus directory...")
    all_files = sorted(corpus_dir.glob("*.txt"))
    all_sids = [p.stem for p in all_files if p.stem.isdigit()]
    print(f"  Found {len(all_sids)} .txt files on disk")

    # Anything on disk not tracked by either CSV = untracked pool item.
    # This is what demoted authors' snippets become — they stay as corpus
    # members but carry no gold/query status.
    untracked = [sid for sid in all_sids if sid not in snippet_author]
    if untracked:
        print(f"  {len(untracked)} untracked .txt files → treated as pool-only")
        for sid in untracked:
            snippet_author[sid] = f"untracked_{sid}"
            snippet_post[sid]   = ""
            snippet_url[sid]    = ""
            is_gold[sid]        = False

    # Anything tracked but missing from disk → drop
    tracked_ids = set(snippet_author)
    on_disk_ids = set(all_sids)
    missing = tracked_ids - on_disk_ids
    if missing:
        print(f"  [!] {len(missing)} tracked snippets missing from disk (first 5: {list(missing)[:5]})")
        for sid in missing:
            snippet_author.pop(sid, None)
            snippet_post.pop(sid, None)
            snippet_url.pop(sid, None)
            is_gold.pop(sid, None)

    print(f"\n  Final corpus composition:")
    print(f"    Gold (tracked in corpus_track.csv):      {sum(1 for v in is_gold.values() if v)}")
    print(f"    Tracked distractors (distractor_track):  {sum(1 for sid in snippet_author if not is_gold[sid] and not snippet_author[sid].startswith('untracked_'))}")
    print(f"    Untracked pool items (demoted/extra):    {sum(1 for a in snippet_author.values() if a.startswith('untracked_'))}")
    print(f"    TOTAL corpus:                            {len(snippet_author)}")

    # ── Determine query authors ──────────────────────────────────────────────
    author_gold_counts = defaultdict(int)
    for sid, a in snippet_author.items():
        if is_gold.get(sid):
            author_gold_counts[a] += 1

    query_authors = {a for a, c in author_gold_counts.items() if c >= args.min_per_author}
    print(f"\n[3/5] Query authors with >= {args.min_per_author} gold snippets: {len(query_authors)}")

    n_query_gold  = sum(1 for sid, a in snippet_author.items()
                        if is_gold.get(sid) and a in query_authors)
    n_pool_gold   = sum(1 for sid, a in snippet_author.items()
                        if is_gold.get(sid) and a not in query_authors)
    print(f"  Snippets by query authors (will be queries + their gold): {n_query_gold}")
    print(f"  Snippets by pool-only gold authors:                       {n_pool_gold}")

    # ── Write corpus.jsonl (every snippet) ───────────────────────────────────
    print("\n[4/5] Writing output files...")
    corpus_path = out_dir / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for sid in sorted(snippet_author, key=lambda x: int(x) if x.isdigit() else x):
            text = read_snippet(corpus_dir / f"{sid}.txt")
            f.write(json.dumps({"_id": sid, "text": text}) + "\n")
    print(f"  {corpus_path}  ({len(snippet_author)} entries)")

    # ── Write queries.jsonl (only gold snippets by query authors) ────────────
    queries_path = out_dir / "queries.jsonl"
    query_ids = []
    with open(queries_path, "w") as f:
        for sid in sorted(snippet_author, key=lambda x: int(x) if x.isdigit() else x):
            if not is_gold.get(sid):
                continue
            if snippet_author[sid] not in query_authors:
                continue
            text = read_snippet(corpus_dir / f"{sid}.txt")
            f.write(json.dumps({"_id": sid, "text": text}) + "\n")
            query_ids.append(sid)
    print(f"  {queries_path}  ({len(query_ids)} queries)")

    # ── Build (author, post) → set of snippet_ids for same-post exclusion ───
    author_snippets  = defaultdict(list)   # author → [sid, ...] (gold only)
    post_snippets    = defaultdict(list)   # (author, post) → [sid, ...] (gold only)
    for sid, a in snippet_author.items():
        if is_gold.get(sid):
            author_snippets[a].append(sid)
            post_snippets[(a, snippet_post[sid])].append(sid)

    # ── qrels.tsv ────────────────────────────────────────────────────────────
    qrels_path = out_dir / "qrels.tsv"
    n_pairs = 0
    with open(qrels_path, "w") as f:
        f.write("query_id\tcorpus_id\tscore\n")
        for qid in query_ids:
            author = snippet_author[qid]
            post   = snippet_post[qid]
            # Gold = all other gold snippets by same author, not in same post
            for other_sid in author_snippets[author]:
                if other_sid == qid:
                    continue
                if snippet_post[other_sid] == post:
                    continue  # same-post exclusion: these aren't gold either
                f.write(f"{qid}\t{other_sid}\t1\n")
                n_pairs += 1
    print(f"  {qrels_path}  ({n_pairs} qrel pairs)")

    # ── per_query_excluded_ids.json ──────────────────────────────────────────
    excluded_path = out_dir / "per_query_excluded_ids.json"
    excluded = {}
    for qid in query_ids:
        author = snippet_author[qid]
        post   = snippet_post[qid]
        # Exclude: the query itself + any same-(author, post) snippets
        ex = set(post_snippets[(author, post)])  # includes qid
        excluded[qid] = sorted(ex)
    with open(excluded_path, "w") as f:
        json.dump(excluded, f, indent=2)
    print(f"  {excluded_path}  ({len(excluded)} queries)")

    # ── Per-query stats ──────────────────────────────────────────────────────
    print("\n[5/5] Per-query gold distribution:")
    gold_counts = []
    for qid in query_ids:
        n = sum(1 for line in open(qrels_path) if line.startswith(f"{qid}\t"))
        gold_counts.append(n)
    if gold_counts:
        gold_counts.sort()
        print(f"  min:    {gold_counts[0]}")
        print(f"  median: {gold_counts[len(gold_counts)//2]}")
        print(f"  mean:   {sum(gold_counts)/len(gold_counts):.1f}")
        print(f"  max:    {gold_counts[-1]}")
        n_zero = sum(1 for c in gold_counts if c == 0)
        if n_zero:
            print(f"  [!] {n_zero} queries have 0 gold (all their other gold is same-post)")

    print("\nDone.")


if __name__ == "__main__":
    main()
