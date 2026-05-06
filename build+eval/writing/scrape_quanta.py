"""
scrape_quanta.py
================
Scrape Quanta Magazine articles by paginating through the archive.

Workflow:
  1. For i in 1..--max_pages: fetch https://www.quantamagazine.org/archive/page/<i>/
  2. Parse article URLs (ends in -YYYYMMDD/) + titles + author from each page.
  3. Fetch each article page, extract body with trafilatura.
  4. Chunk into snippets, emit JSONL.

Each record: {source: "quanta", author, url, title, date, text}.

Usage:
  python scrape_quanta.py --out quanta_articles.jsonl --max_pages 300
  python scrape_quanta.py --out quanta_articles.jsonl --max_pages 10  # test
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


BASE_URL   = "https://www.quantamagazine.org/archive/page/{i}/"
ARTICLE_RE = re.compile(r"^https://www\.quantamagazine\.org/[a-z0-9\-]+-(\d{8})/?$")
USER_AGENT = "Mozilla/5.0 (research scraper; academic use; Writing Analogues benchmark v2)"


def fetch(url, retries=3, sleep=1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                time.sleep(sleep * (2 ** attempt) + 2)
                continue
            if r.status_code == 404:
                return None  # exhausted archive
            return None
        except requests.RequestException:
            time.sleep(sleep * (2 ** attempt))
    return None


def parse_archive_page(html):
    """Return list of (url, title, date) for articles on this page."""
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = "https://www.quantamagazine.org" + href
        m = ARTICLE_RE.match(href)
        if not m:
            continue
        if href in seen:
            continue
        seen.add(href)
        title = a.get_text(strip=True)
        if not title or len(title) < 5:
            # Anchor with an image — need to look nearby for title. Try the
            # next sibling or find an h-tag in the link's ancestor chain.
            parent = a.find_parent()
            if parent:
                h = parent.find(["h2", "h3"])
                if h:
                    title = h.get_text(strip=True)
        if title and len(title) > 5:
            date = m.group(1)  # YYYYMMDD
            out.append((href, title, date))
    return out


def extract_article(html):
    """Return (text, author) from article page."""
    if not html:
        return "", None
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    # Author: Quanta article pages have meta author tags; try a few patterns.
    author = None
    soup = BeautifulSoup(html, "html.parser")
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author:
        author = meta_author.get("content", "").strip()
    if not author:
        # Fallback: <a class="byline__author"> or similar
        link = soup.find("a", href=re.compile(r"/authors/"))
        if link:
            author = link.get_text(strip=True)
    return (text or ""), author


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",       default="quanta_articles.jsonl")
    p.add_argument("--max_pages", type=int, default=300,
                   help="Pagination depth on /archive/page/<i>/")
    p.add_argument("--start_page", type=int, default=1)
    p.add_argument("--sleep",      type=float, default=0.8)
    p.add_argument("--min_words",  type=int, default=200)
    p.add_argument("--max_words",  type=int, default=850)
    p.add_argument("--max_per_article", type=int, default=2)
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for per-snippet length sampling")
    p.add_argument("--resume", action="store_true",
                   help="Skip URLs already present in --out")
    args = p.parse_args()

    rng = random.Random(args.seed)

    done_urls = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_urls.add(json.loads(line)["url"])
                except Exception:
                    pass
        print(f"Resume: {len(done_urls)} URLs already scraped")

    # Phase 1: collect all article URLs from archive pagination
    all_articles = []
    print(f"Phase 1: paginating archive pages {args.start_page}..{args.max_pages}")
    for i in tqdm(range(args.start_page, args.max_pages + 1), desc="pages"):
        html = fetch(BASE_URL.format(i=i))
        if not html:
            print(f"\n  page {i} returned nothing — stopping pagination")
            break
        page_articles = parse_archive_page(html)
        if not page_articles:
            print(f"\n  page {i} had 0 parseable articles — stopping")
            break
        all_articles.extend(page_articles)
        time.sleep(args.sleep)

    # Dedupe
    seen = set()
    uniq = []
    for url, title, date in all_articles:
        if url not in seen and url not in done_urls:
            seen.add(url)
            uniq.append((url, title, date))
    print(f"Phase 1 done: {len(uniq)} unique articles to fetch")

    # Phase 2: fetch each article, extract, chunk
    mode = "a" if args.resume else "w"
    n_snips = 0
    n_articles = 0
    with open(out_path, mode) as f:
        for url, title, date in tqdm(uniq, desc="articles"):
            html = fetch(url)
            text, author = extract_article(html)
            if not text or len(text) < 400:
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
                    "source":     "quanta",
                    "url":        url,
                    "title":      title,
                    "author":     author or "unknown",
                    "date":       date,
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
