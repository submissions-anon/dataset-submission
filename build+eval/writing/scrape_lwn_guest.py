"""
scrape_lwn_guest.py
===================
Scrape LWN.net guest articles from the guest author index.

Workflow:
  1. Fetch https://lwn.net/Archives/GuestIndex/ → parse author sections.
  2. Extract every article URL listed under each author.
  3. Fetch each article page, extract body via trafilatura.
  4. Chunk into snippets, emit as JSONL.

Each record: {source: "lwn", article_id: <LWN internal number>, author: <name>, url, title, text_snippet}.

Usage:
  python scrape_lwn_guest.py --out lwn_articles.jsonl
  python scrape_lwn_guest.py --out lwn_articles.jsonl --limit 50
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm

from chunker import extract_snippets


INDEX_URL   = "https://lwn.net/Archives/GuestIndex/"
ARTICLE_RE  = re.compile(r"/Articles/(\d+)/?$")
USER_AGENT  = "Mozilla/5.0 (research scraper; academic use; Writing Analogues benchmark v2)"


def fetch(url, retries=3, sleep=1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                time.sleep(sleep * (2 ** attempt) + 2)
                continue
            return None
        except requests.RequestException:
            time.sleep(sleep * (2 ** attempt))
    return None


def parse_index(html):
    """Return list of (author, article_id, article_url, title).

    LWN's guest index groups articles by author in a flat list where each
    author is followed by their articles. We walk the DOM linearly and
    track the current author.
    """
    soup = BeautifulSoup(html, "html.parser")
    out = []
    current_author = None

    # The page uses <a name="Author_Name"> anchors for authors, followed by
    # <a href="/Articles/NNNNNN/">title</a> for articles. We scan all <a>
    # tags in document order.
    for a in soup.find_all("a"):
        name_anchor = a.get("name")
        if name_anchor:
            # Author anchor like <a name="Aurora_Valerie">
            # The display name is the text of the previous <h3> or the next
            # link text; simplest: convert anchor to "Valerie Aurora" form.
            parts = name_anchor.split("_")
            if len(parts) >= 2:
                # "Aurora_Valerie" -> "Valerie Aurora"
                current_author = " ".join(parts[1:] + [parts[0]])
            else:
                current_author = parts[0]
            continue

        href = a.get("href", "")
        m = ARTICLE_RE.search(href)
        if m and current_author:
            article_id = m.group(1)
            full_url = f"https://lwn.net/Articles/{article_id}/"
            title = a.get_text(strip=True)
            if title and len(title) > 3:
                out.append((current_author, article_id, full_url, title))

    return out


def extract_article_body(html):
    """Use trafilatura to pull main text out of an LWN article page."""
    if not html:
        return ""
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    return text or ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",       default="lwn_articles.jsonl")
    p.add_argument("--limit",     type=int, default=None,
                   help="Max number of articles to fetch (for testing)")
    p.add_argument("--sleep",     type=float, default=1.0)
    p.add_argument("--min_words", type=int, default=200)
    p.add_argument("--max_words", type=int, default=850)
    p.add_argument("--max_per_article", type=int, default=2)
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for per-snippet length sampling")
    p.add_argument("--resume", action="store_true",
                   help="Skip article_ids already present in --out")
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"Fetching LWN guest article index from {INDEX_URL}")
    index_html = fetch(INDEX_URL)
    if not index_html:
        raise SystemExit("Failed to fetch guest index")

    entries = parse_index(index_html)
    print(f"Parsed {len(entries)} articles across {len(set(e[0] for e in entries))} authors")

    if args.limit:
        entries = entries[: args.limit]
        print(f"Limiting to {len(entries)} articles")

    # Resume support
    done_ids = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["article_id"])
                except Exception:
                    pass
        print(f"Resume: {len(done_ids)} article_ids already scraped")

    mode = "a" if args.resume else "w"
    n_snips = 0
    n_articles = 0
    with open(out_path, mode) as f:
        for author, aid, url, title in tqdm(entries, desc="LWN"):
            if aid in done_ids:
                continue

            html = fetch(url)
            text = extract_article_body(html)
            if not text or len(text) < 300:
                continue

            snips = extract_snippets(
                text,
                target_min=args.min_words,
                target_max=args.max_words,
                max_per_article=args.max_per_article,
                rng=rng,
            )
            for s in snips:
                rec = {
                    "source":     "lwn",
                    "article_id": aid,
                    "author":     author,
                    "url":        url,
                    "title":      title,
                    "text":       s,
                    "word_count": len(s.split()),
                }
                f.write(json.dumps(rec) + "\n")
                n_snips += 1
            f.flush()
            n_articles += 1
            time.sleep(args.sleep)

    print(f"\nDone. {n_articles} articles → {n_snips} snippets → {out_path}")


if __name__ == "__main__":
    main()
