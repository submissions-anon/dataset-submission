"""
scrape_lesswrong.py
===================
Scrape LessWrong Personal Blogposts via GraphQL. Bypasses the infinite-scroll
UI entirely — the GraphQL endpoint returns paginated post batches directly.

Filters:
  - view = "new" or "top" (we use "new" for temporal coverage)
  - filter = "personal" (Personal Blogposts only, NOT Frontpage/Curated)
  - karma threshold (default 10 to avoid very low-effort posts)
  - min word count (default 500 for non-trivial content)
  - gold-author blocklist (exclude posts by authors in your gold list
    so they don't leak into distractor pool)

Output JSONL, same schema as scrape_quanta.py / scrape_lwn_guest.py.

Usage:
  export LW_GOLD_BLOCKLIST="Jacob Steinhardt,Sarah Constantin,Ege Erdil,..."
  python scrape_lesswrong.py --out lw_posts.jsonl --n_posts 3000
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from chunker import extract_snippets

GRAPHQL_URL = "https://www.lesswrong.com/graphql"
USER_AGENT  = "Mozilla/5.0 (research scraper; academic use; Writing Analogues benchmark v2)"


POSTS_QUERY = """
query getPosts($terms: JSON, $enableCache: Boolean) {
  posts(input: {terms: $terms, enableCache: $enableCache}) {
    results {
      _id
      slug
      title
      pageUrl
      postedAt
      baseScore
      wordCount
      user { displayName slug }
      coauthors { displayName }
      contents { plaintextMainText }
    }
  }
}
"""

# LW's GraphQL caps offset pagination at skip<=2000. To reach more posts we
# paginate by time: each batch asks for posts before the earliest postedAt
# of the previous batch.


def graphql_query(query, variables, retries=3, sleep=1.5):
    payload = {"query": query, "variables": variables}
    headers = {
        "Content-Type": "application/json",
        "User-Agent":   USER_AGENT,
    }
    for attempt in range(retries):
        try:
            r = requests.post(GRAPHQL_URL, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if "errors" in data:
                    # Some errors are fatal (schema mismatch), some transient
                    print(f"  GraphQL errors: {data['errors']}")
                    if attempt < retries - 1:
                        time.sleep(sleep * (2 ** attempt))
                        continue
                return data
            if r.status_code == 429:
                time.sleep(sleep * (2 ** attempt) + 3)
                continue
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"  RequestException: {e}")
        time.sleep(sleep * (2 ** attempt))
    return None


def fetch_posts_batch(limit, karma_threshold, view="new", before=None):
    """One page of posts matching our filters, optionally before a given
    postedAt timestamp (ISO 8601). This bypasses the skip<=2000 cap."""
    terms = {
        "view":           view,
        "filter":         "personal",
        "karmaThreshold": karma_threshold,
        "limit":          limit,
    }
    if before:
        terms["before"] = before
    data = graphql_query(POSTS_QUERY, {"terms": terms, "enableCache": False})
    if not data:
        return []
    try:
        return data["data"]["posts"]["results"] or []
    except (KeyError, TypeError):
        return []


def normalize_author(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",             default="lw_posts.jsonl")
    p.add_argument("--n_posts",         type=int, default=3000,
                   help="Max number of posts to try to fetch")
    p.add_argument("--karma_threshold", type=int, default=10)
    p.add_argument("--min_words",       type=int, default=500,
                   help="Minimum post word count (before snippet extraction)")
    p.add_argument("--view",            choices=["new", "top"], default="new")
    p.add_argument("--batch_size",      type=int, default=50,
                   help="GraphQL page size")
    p.add_argument("--sleep",           type=float, default=1.5)
    p.add_argument("--gold_blocklist",  default=None,
                   help="Comma-separated list of gold-author names to exclude. "
                        "If omitted, reads LW_GOLD_BLOCKLIST env var.")
    p.add_argument("--snip_min",        type=int, default=200)
    p.add_argument("--snip_max",        type=int, default=850)
    p.add_argument("--max_per_article", type=int, default=2)
    p.add_argument("--rng_seed",        type=int, default=42,
                   help="RNG seed for per-snippet length sampling")
    p.add_argument("--resume",          action="store_true")
    args = p.parse_args()

    rng = random.Random(args.rng_seed)

    # Blocklist
    raw_blocklist = args.gold_blocklist or os.environ.get("LW_GOLD_BLOCKLIST", "")
    blocklist = {normalize_author(x) for x in raw_blocklist.split(",") if x.strip()}
    if blocklist:
        print(f"Gold-author blocklist: {len(blocklist)} names excluded")
        for name in sorted(blocklist):
            print(f"  - {name}")
    else:
        print("WARNING: no gold-author blocklist set. "
              "Distractor pool may contain posts by gold authors.")

    # Resume: if file exists, also track the earliest postedAt we've seen
    # so we can continue from before that timestamp.
    out_path = Path(args.out)
    done_ids = set()
    earliest_posted_at = None  # most recent "before" cursor we should start with
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["post_id"])
                    pa = rec.get("posted_at")
                    if pa and (earliest_posted_at is None or pa < earliest_posted_at):
                        earliest_posted_at = pa
                except Exception:
                    pass
        print(f"Resume: {len(done_ids)} post_ids already scraped; "
              f"resuming before postedAt={earliest_posted_at}")

    mode = "a" if args.resume else "w"
    before_cursor = earliest_posted_at  # None → start from newest
    fetched = 0
    n_snips = 0
    n_posts_kept = 0
    n_blocked = 0
    consecutive_empty = 0
    pbar = tqdm(total=args.n_posts, desc="LW posts")
    with open(out_path, mode) as f:
        while fetched < args.n_posts:
            batch = fetch_posts_batch(
                limit=args.batch_size,
                karma_threshold=args.karma_threshold,
                view=args.view,
                before=before_cursor,
            )
            if not batch:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    print(f"\n  two empty batches in a row — stopping")
                    break
                time.sleep(args.sleep * 2)
                continue
            consecutive_empty = 0

            # Track oldest postedAt in this batch to use as next cursor
            batch_oldest = None

            for post in batch:
                fetched += 1
                pbar.update(1)
                if fetched >= args.n_posts:
                    break

                posted_at = post.get("postedAt")
                if posted_at and (batch_oldest is None or posted_at < batch_oldest):
                    batch_oldest = posted_at

                post_id = post.get("_id")
                if post_id in done_ids:
                    continue

                title = (post.get("title") or "").strip()
                url   = post.get("pageUrl") or f"https://www.lesswrong.com/posts/{post_id}"
                word_count = post.get("wordCount") or 0
                if word_count < args.min_words:
                    continue

                user = post.get("user") or {}
                author = (user.get("displayName") or "").strip()
                coauthors = [c.get("displayName", "") for c in (post.get("coauthors") or [])]
                all_author_names = [author] + coauthors
                if any(normalize_author(a) in blocklist for a in all_author_names if a):
                    n_blocked += 1
                    continue
                if not author:
                    continue

                contents = post.get("contents") or {}
                text = contents.get("plaintextMainText") or ""
                if len(text) < 500:
                    continue

                snips = extract_snippets(
                    text,
                    target_min=args.snip_min,
                    target_max=args.snip_max,
                    max_per_article=args.max_per_article,
                    rng=rng,
                )
                for s in snips:
                    rec = {
                        "source":     "lesswrong",
                        "post_id":    post_id,
                        "author":     author,
                        "url":        url,
                        "title":      title,
                        "posted_at":  posted_at,
                        "karma":      post.get("baseScore"),
                        "text":       s,
                        "word_count": len(s.split()),
                    }
                    f.write(json.dumps(rec) + "\n")
                    n_snips += 1
                f.flush()
                if snips:
                    n_posts_kept += 1

            if batch_oldest is None:
                # Something off — bail to avoid infinite loop
                print(f"\n  batch had no timestamps — stopping")
                break
            # Advance cursor past the oldest post in this batch
            before_cursor = batch_oldest
            time.sleep(args.sleep)

    pbar.close()
    print(f"\nDone. Fetched {fetched} posts, kept {n_posts_kept}, blocked {n_blocked} by blocklist.")
    print(f"Wrote {n_snips} snippets → {out_path}")


if __name__ == "__main__":
    main()
