"""
Congressional Hearing Corpus Builder for Tip-of-Tongue (ToT) Benchmark
=====================================================================

Scrapes congressional hearing transcripts from the GovInfo API,
segments them into coherent Q&A exchanges at speaker-turn boundaries,
and outputs in BEIR format (corpus.jsonl, queries.jsonl, qrels.tsv).

Usage:
    1. Get a free API key from https://api.data.gov/
    2. export GOVINFO_API_KEY=your_key_here
    3. python build_congress_corpus.py --phase list    # discover hearings
    4. python build_congress_corpus.py --phase download # fetch HTML
    5. python build_congress_corpus.py --phase segment  # chunk into passages
    6. python build_congress_corpus.py --phase export   # BEIR format

Each phase is resume-safe with checkpoint files.
"""

import argparse
import asyncio
import json
import os
import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from html.parser import HTMLParser

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

API_BASE = "https://api.govinfo.gov"
API_KEY = os.environ.get("GOVINFO_API_KEY", "DEMO_KEY")

DATA_DIR = Path("congress_corpus_data")
LIST_DIR = DATA_DIR / "listings"
HTML_DIR = DATA_DIR / "html"
SEGMENT_DIR = DATA_DIR / "segments"
EXPORT_DIR = DATA_DIR / "beir_export"

# Committees most likely to have tech-adjacent testimony
# (broad enough to generate thematic distractors)
TARGET_COMMITTEES = [
    "commerce",        # Senate Commerce, Science & Transportation
    "judiciary",       # Senate/House Judiciary (antitrust, privacy)
    "energy",          # House Energy and Commerce
    "science",         # House Science, Space, and Technology
    "intelligence",    # Senate/House Intelligence
    "finance",         # Senate Finance
    "banking",         # Senate Banking (fintech, crypto)
    "oversight",       # House Oversight
    "appropriations",  # for volume / distractors
    "armed",           # for volume / distractors
    "health",          # pharma hearings (rhetorical pattern overlap)
    "education",       # for volume
    "homeland",        # cybersecurity hearings
    "foreign",         # for volume
    "budget",          # for volume
    "small business",  # tech platform impact
    "antitrust",       # subcommittee keyword
]

# Congress range: 110th (2007) through 118th (2023-2024)
# gives us ~17 years of hearings including all major tech hearings
CONGRESS_START = 110
CONGRESS_END = 119

# Rate limiting
REQUESTS_PER_SECOND = 10  # well under the 40/s limit
SEMAPHORE_LIMIT = 5

# Target corpus size
TARGET_PASSAGES = 200_000


# ─────────────────────────────────────────────────────────
# Phase 1: List available hearing packages
# ─────────────────────────────────────────────────────────

async def list_hearings():
    """
    Use the collections endpoint to enumerate all CHRG packages.
    We paginate through the full collection within our congress range.
    """
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = LIST_DIR / "package_ids.jsonl"

    # Load existing checkpoint
    existing_ids = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                rec = json.loads(line)
                existing_ids.add(rec["packageId"])
        logger.info(f"Resuming from checkpoint: {len(existing_ids)} packages already listed")

    # The collections endpoint needs a start date
    # 110th Congress started Jan 2007
    start_date = "2007-01-01T00:00:00Z"

    all_packages = []
    offset_mark = "*"
    page_size = 1000

    async with aiohttp.ClientSession() as session:
        while True:
            url = (
                f"{API_BASE}/collections/CHRG/{start_date}"
                f"?offsetMark={offset_mark}&pageSize={page_size}"
                f"&api_key={API_KEY}"
            )
            logger.info(f"Fetching collection page (offsetMark={offset_mark[:20]}...)")

            async with session.get(url) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status != 200:
                    logger.error(f"HTTP {resp.status}: {await resp.text()}")
                    break

                data = await resp.json()

            packages = data.get("packages", [])
            if not packages:
                logger.info("No more packages, done listing.")
                break

            # Filter by congress range based on packageId format: CHRG-{congress}{chamber}...
            for pkg in packages:
                pid = pkg.get("packageId", "")
                # Extract congress number from packageId
                # Format: CHRG-116shrg12345 or CHRG-116hhrg12345
                match = re.match(r"CHRG-(\d+)", pid)
                if match:
                    congress_num = int(match.group(1))
                    if CONGRESS_START <= congress_num <= CONGRESS_END:
                        if pid not in existing_ids:
                            record = {
                                "packageId": pid,
                                "lastModified": pkg.get("lastModified", ""),
                                "packageLink": pkg.get("packageLink", ""),
                            }
                            all_packages.append(record)
                            existing_ids.add(pid)

            # Get next page
            next_page = data.get("nextPage")
            if not next_page:
                break
            # Extract offsetMark from nextPage URL
            import urllib.parse
            parsed = urllib.parse.urlparse(next_page)
            params = urllib.parse.parse_qs(parsed.query)
            offset_mark = params.get("offsetMark", [""])[0]
            if not offset_mark:
                break

            await asyncio.sleep(1.0 / REQUESTS_PER_SECOND)

    # Append new packages to checkpoint
    if all_packages:
        with open(checkpoint_file, "a") as f:
            for pkg in all_packages:
                f.write(json.dumps(pkg) + "\n")
        logger.info(f"Listed {len(all_packages)} new packages (total: {len(existing_ids)})")
    else:
        logger.info(f"No new packages found (total: {len(existing_ids)})")


# ─────────────────────────────────────────────────────────
# Phase 2: Download HTML content + metadata
# ─────────────────────────────────────────────────────────

async def download_hearing(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                           package_id: str, meta_dir: Path, html_dir: Path):
    """Download summary metadata and HTML text for a single hearing."""
    meta_file = meta_dir / f"{package_id}.json"
    html_file = html_dir / f"{package_id}.html"

    # Skip if already downloaded
    if meta_file.exists() and html_file.exists():
        return "skip"

    async with sem:
        # 1. Get summary metadata
        if not meta_file.exists():
            url = f"{API_BASE}/packages/{package_id}/summary?api_key={API_KEY}"
            try:
                async with session.get(url) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        await asyncio.sleep(retry_after)
                        return "retry"
                    if resp.status != 200:
                        logger.warning(f"Summary {resp.status} for {package_id}")
                        return "error"
                    meta = await resp.json()
                    with open(meta_file, "w") as f:
                        json.dump(meta, f)
            except Exception as e:
                logger.warning(f"Error fetching summary for {package_id}: {e}")
                return "error"

            await asyncio.sleep(1.0 / REQUESTS_PER_SECOND)

        # 2. Check if this hearing is from a relevant committee
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            # Committee info is in various fields depending on the package
            committee_str = json.dumps(meta).lower()
            is_relevant = any(kw in committee_str for kw in TARGET_COMMITTEES)
            if not is_relevant:
                # Still download — we want volume for distractors
                # But mark it so we can prioritize later
                pass

        # 3. Get HTML content
        if not html_file.exists():
            url = f"{API_BASE}/packages/{package_id}/htm?api_key={API_KEY}"
            try:
                async with session.get(url) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        await asyncio.sleep(retry_after)
                        return "retry"
                    if resp.status == 404:
                        # Some packages don't have HTML, try text
                        url_txt = f"{API_BASE}/packages/{package_id}/txt?api_key={API_KEY}"
                        async with session.get(url_txt) as resp2:
                            if resp2.status == 200:
                                text = await resp2.text()
                                with open(html_file, "w") as f:
                                    f.write(text)
                                return "ok"
                        logger.warning(f"No HTML/TXT for {package_id}")
                        # Write empty marker so we don't retry
                        html_file.write_text("")
                        return "no_content"
                    if resp.status != 200:
                        logger.warning(f"HTML {resp.status} for {package_id}")
                        return "error"
                    text = await resp.text()
                    with open(html_file, "w") as f:
                        f.write(text)
            except Exception as e:
                logger.warning(f"Error fetching HTML for {package_id}: {e}")
                return "error"

            await asyncio.sleep(1.0 / REQUESTS_PER_SECOND)

    return "ok"


async def download_hearings():
    """Download all listed hearings."""
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    meta_dir = DATA_DIR / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    listing_file = LIST_DIR / "package_ids.jsonl"
    if not listing_file.exists():
        logger.error("No listing file found. Run --phase list first.")
        return

    package_ids = []
    with open(listing_file) as f:
        for line in f:
            rec = json.loads(line)
            package_ids.append(rec["packageId"])

    logger.info(f"Downloading {len(package_ids)} hearings...")

    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    stats = {"ok": 0, "skip": 0, "error": 0, "retry": 0, "no_content": 0}

    async with aiohttp.ClientSession() as session:
        # Process in batches to avoid overwhelming
        batch_size = 50
        for i in range(0, len(package_ids), batch_size):
            batch = package_ids[i:i + batch_size]
            tasks = [
                download_hearing(session, sem, pid, meta_dir, HTML_DIR)
                for pid in batch
            ]
            results = await asyncio.gather(*tasks)
            for r in results:
                stats[r] = stats.get(r, 0) + 1

            logger.info(
                f"Batch {i // batch_size + 1}: "
                f"ok={stats['ok']} skip={stats['skip']} "
                f"error={stats['error']} no_content={stats['no_content']}"
            )

    logger.info(f"Download complete: {stats}")


# ─────────────────────────────────────────────────────────
# Phase 3: Segment into coherent passages
# ─────────────────────────────────────────────────────────

class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags, keep text."""
    def __init__(self):
        super().__init__()
        self.result = []

    def handle_data(self, data):
        self.result.append(data)

    def get_text(self):
        return "".join(self.result)


def strip_html(html_text: str) -> str:
    """Remove HTML tags, return plain text."""
    extractor = HTMLTextExtractor()
    extractor.feed(html_text)
    return extractor.get_text()


# Patterns that mark speaker turns in hearing transcripts
# These handle various transcript formats:
#   "Senator WARREN. Thank you..."
#   "Mr. ZUCKERBERG. Thank you..."
#   "The CHAIRMAN. The committee will..."
#   "Rep. Cicilline: (00:00) ..."
SPEAKER_PATTERNS = [
    # Official GPO format: Title LASTNAME.
    re.compile(
        r"^[ \t]*(Senator|Chairman|Chairwoman|Chairperson|"
        r"Mr\.|Mrs\.|Ms\.|Dr\.|Representative|Rep\.|"
        r"The CHAIRMAN|The CHAIRWOMAN|The CHAIR|"
        r"General|Admiral|Secretary|Commissioner|Director|"
        r"Ambassador|Governor|Mayor|Professor|Judge|Justice)"
        r"[.\s]+([A-Z][A-Za-z\-\']+(?:\s+[A-Z][A-Za-z\-\']+)?)\s*[.\:]\s*",
        re.MULTILINE
    ),
    # Rev.com / unofficial format: "Name: (timestamp)"
    re.compile(
        r"^[ \t]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*:\s*(?:\(\d+:\d+\))?\s*",
        re.MULTILINE
    ),
    # All-caps name format: "ZUCKERBERG."
    re.compile(
        r"^[ \t]*([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*[.]\s*",
        re.MULTILINE
    ),
]


@dataclass
class Passage:
    doc_id: str
    text: str
    speaker: str
    hearing_id: str
    hearing_title: str
    congress: int
    committee: str
    passage_type: str  # "opening_statement", "qa_exchange", "colloquy", "procedural"
    word_count: int


def identify_speaker_turns(text: str) -> list[dict]:
    """
    Split transcript text into speaker turns.
    Returns list of {speaker, text, start_pos}.
    """
    # Try each pattern, use whichever finds the most matches
    best_splits = []
    for pattern in SPEAKER_PATTERNS:
        matches = list(pattern.finditer(text))
        if len(matches) > len(best_splits):
            best_splits = matches

    if not best_splits:
        # No speaker turns found — treat entire text as one block
        return [{"speaker": "UNKNOWN", "text": text.strip(), "start": 0}]

    turns = []
    for i, match in enumerate(best_splits):
        start = match.end()
        end = best_splits[i + 1].start() if i + 1 < len(best_splits) else len(text)
        speaker_text = text[start:end].strip()

        # Extract speaker name from match
        groups = match.groups()
        speaker = " ".join(g for g in groups if g).strip(". ")

        if speaker_text:  # skip empty turns
            turns.append({
                "speaker": speaker,
                "text": speaker_text,
                "start": match.start(),
            })

    return turns


def group_into_exchanges(turns: list[dict]) -> list[list[dict]]:
    """
    Group speaker turns into coherent exchanges.

    An exchange is:
    - A questioner (Senator/Rep) + respondent (witness) sequence
    - Kept together until a NEW questioner starts
    - Opening statements kept as standalone passages

    This preserves the natural coherence of the Q&A format.
    """
    if not turns:
        return []

    # Classify speakers as questioners (members of Congress) vs witnesses
    # Key insight: In congressional transcripts, members are addressed as
    # "Senator X", "Chairman X", "Rep. X", or "The CHAIRMAN".
    # Witnesses are "Mr. X", "Mrs. X", "Ms. X", "Dr. X", "General X", etc.
    # So Mr./Mrs./Ms./Dr. are WITNESS indicators, not questioner indicators.
    questioner_titles = {
        "senator", "chairman", "chairwoman", "chairperson",
        "the chairman", "the chairwoman", "the chair",
        "representative", "rep",
    }

    def is_questioner(speaker: str) -> bool:
        s = speaker.lower().strip()
        for title in questioner_titles:
            if s.startswith(title):
                return True
        return False

    exchanges = []
    current_exchange = []

    for turn in turns:
        if is_questioner(turn["speaker"]) and current_exchange:
            # New questioner = start of new exchange
            # But only if current exchange already has a questioner
            has_questioner = any(is_questioner(t["speaker"]) for t in current_exchange)
            if has_questioner:
                exchanges.append(current_exchange)
                current_exchange = []

        current_exchange.append(turn)

    if current_exchange:
        exchanges.append(current_exchange)

    return exchanges


def classify_passage(exchange: list[dict]) -> str:
    """Classify the type of an exchange."""
    if len(exchange) == 1:
        word_count = len(exchange[0]["text"].split())
        if word_count > 300:
            return "opening_statement"
        return "procedural"
    return "qa_exchange"


def segment_hearing(package_id: str, html_text: str, meta: dict) -> list[Passage]:
    """
    Segment a single hearing transcript into coherent passages.
    """
    # Strip HTML
    text = strip_html(html_text)

    if len(text.strip()) < 100:
        return []

    # Extract metadata
    title = meta.get("title", "Unknown Hearing")
    congress_match = re.match(r"CHRG-(\d+)", package_id)
    congress = int(congress_match.group(1)) if congress_match else 0

    committees = meta.get("committees", [])
    committee_str = ", ".join(
        c.get("committeeName", "") for c in committees
    ) if isinstance(committees, list) else str(committees)

    # Split into speaker turns
    turns = identify_speaker_turns(text)

    # Group into exchanges
    exchanges = group_into_exchanges(turns)

    passages = []
    for idx, exchange in enumerate(exchanges):
        # Combine all turns in the exchange into one passage
        combined_text = "\n\n".join(
            f"{t['speaker']}: {t['text']}" for t in exchange
        )

        word_count = len(combined_text.split())

        # Skip very short passages (procedural noise)
        if word_count < 50:
            continue

        # If passage is very long (>1000 words), it's likely a prepared
        # statement or multiple exchanges that weren't split well.
        # Split at ~500 word boundaries while respecting sentence boundaries.
        if word_count > 1000:
            sub_passages = split_long_passage(combined_text, max_words=500)
            for sub_idx, sub_text in enumerate(sub_passages):
                doc_id = f"{package_id}_p{idx:04d}_{sub_idx:02d}"
                passages.append(Passage(
                    doc_id=doc_id,
                    text=sub_text,
                    speaker=exchange[0]["speaker"],
                    hearing_id=package_id,
                    hearing_title=title,
                    congress=congress,
                    committee=committee_str,
                    passage_type=classify_passage(exchange),
                    word_count=len(sub_text.split()),
                ))
        else:
            doc_id = f"{package_id}_p{idx:04d}"
            passages.append(Passage(
                doc_id=doc_id,
                text=combined_text,
                speaker=exchange[0]["speaker"],
                hearing_id=package_id,
                hearing_title=title,
                congress=congress,
                committee=committee_str,
                passage_type=classify_passage(exchange),
                word_count=word_count,
            ))

    return passages


def split_long_passage(text: str, max_words: int = 500) -> list[str]:
    """
    Split a long passage at sentence boundaries, targeting max_words per chunk.
    Ensures each chunk is coherent.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_words + s_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0
        current_chunk.append(sentence)
        current_words += s_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def segment_all_hearings():
    """Process all downloaded hearings into passages."""
    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = SEGMENT_DIR / "segments.jsonl"
    progress_file = SEGMENT_DIR / "progress.json"

    # Load progress
    processed_ids = set()
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
            processed_ids = set(progress.get("processed", []))
        logger.info(f"Resuming: {len(processed_ids)} hearings already segmented")

    meta_dir = DATA_DIR / "metadata"
    total_passages = 0
    new_processed = []

    # Count existing passages
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            total_passages = sum(1 for _ in f)
        logger.info(f"Existing passages: {total_passages}")

    html_files = sorted(HTML_DIR.glob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files to process")

    with open(checkpoint_file, "a") as out_f:
        for html_file in html_files:
            package_id = html_file.stem
            if package_id in processed_ids:
                continue

            # Read HTML
            html_text = html_file.read_text(errors="replace")
            if len(html_text.strip()) < 100:
                processed_ids.add(package_id)
                new_processed.append(package_id)
                continue

            # Read metadata
            meta_file = meta_dir / f"{package_id}.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)

            # Segment
            passages = segment_hearing(package_id, html_text, meta)

            for p in passages:
                out_f.write(json.dumps(asdict(p)) + "\n")
                total_passages += 1

            processed_ids.add(package_id)
            new_processed.append(package_id)

            if len(new_processed) % 50 == 0:
                logger.info(
                    f"Processed {len(new_processed)} new hearings, "
                    f"total passages: {total_passages}"
                )
                # Save progress checkpoint
                with open(progress_file, "w") as pf:
                    json.dump({"processed": list(processed_ids)}, pf)


              
         

    # Final progress save
    with open(progress_file, "w") as pf:
        json.dump({"processed": list(processed_ids)}, pf)

    logger.info(f"Segmentation complete: {total_passages} total passages from {len(processed_ids)} hearings")


# ─────────────────────────────────────────────────────────
# Phase 4: Export to BEIR format
# ─────────────────────────────────────────────────────────

def export_beir():
    """
    Export segmented passages to BEIR format:
    - corpus.jsonl: {"_id": ..., "title": ..., "text": ..., "metadata": {...}}
    - (queries.jsonl and qrels.tsv to be created manually / via GPT-5.2)
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    segments_file = SEGMENT_DIR / "segments.jsonl"
    if not segments_file.exists():
        logger.error("No segments file found. Run --phase segment first.")
        return

    corpus_file = EXPORT_DIR / "corpus.jsonl"
    stats_file = EXPORT_DIR / "corpus_stats.json"

    # Stats tracking
    stats = {
        "total_passages": 0,
        "by_congress": {},
        "by_type": {},
        "by_committee_keyword": {},
        "word_count_distribution": {"<100": 0, "100-200": 0, "200-500": 0, "500+": 0},
        "unique_hearings": set(),
    }

    with open(segments_file) as in_f, open(corpus_file, "w") as out_f:
        for line in in_f:
            passage = json.loads(line)

            # BEIR corpus format
            beir_record = {
                "_id": passage["doc_id"],
                "title": f"{passage['speaker']} — {passage['hearing_title'][:100]}",
                "text": passage["text"],
                "metadata": {
                    "hearing_id": passage["hearing_id"],
                    "congress": passage["congress"],
                    "committee": passage["committee"],
                    "passage_type": passage["passage_type"],
                    "speaker": passage["speaker"],
                    "word_count": passage["word_count"],
                }
            }
            out_f.write(json.dumps(beir_record) + "\n")

            # Update stats
            stats["total_passages"] += 1
            c = str(passage["congress"])
            stats["by_congress"][c] = stats["by_congress"].get(c, 0) + 1
            t = passage["passage_type"]
            stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
            stats["unique_hearings"].add(passage["hearing_id"])

            wc = passage["word_count"]
            if wc < 100:
                stats["word_count_distribution"]["<100"] += 1
            elif wc < 200:
                stats["word_count_distribution"]["100-200"] += 1
            elif wc < 500:
                stats["word_count_distribution"]["200-500"] += 1
            else:
                stats["word_count_distribution"]["500+"] += 1

            # Track committee keywords
            for kw in TARGET_COMMITTEES:
                if kw in passage["committee"].lower():
                    stats["by_committee_keyword"][kw] = (
                        stats["by_committee_keyword"].get(kw, 0) + 1
                    )

    stats["unique_hearings"] = len(stats["unique_hearings"])

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Create placeholder query and qrels files
    queries_file = EXPORT_DIR / "queries.jsonl"
    qrels_file = EXPORT_DIR / "qrels.tsv"

    if not queries_file.exists():
        queries_file.write_text(
            "# Placeholder — generate ToT queries from tech CEO hearings\n"
            "# Format: {\"_id\": \"q1\", \"text\": \"do you remember when...\"}\n"
        )
    if not qrels_file.exists():
        qrels_file.write_text(
            "query-id\tcorpus-id\tscore\n"
            "# Placeholder — annotate after query generation + pooled retrieval\n"
        )

    logger.info(f"BEIR export complete:")
    logger.info(f"  Corpus: {stats['total_passages']} passages from {stats['unique_hearings']} hearings")
    logger.info(f"  By type: {stats['by_type']}")
    logger.info(f"  Word count dist: {stats['word_count_distribution']}")
    logger.info(f"  Output: {EXPORT_DIR}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build congressional hearing corpus for ToT benchmark"
    )
    parser.add_argument(
        "--phase",
        choices=["list", "download", "segment", "export", "all"],
        required=True,
        help="Which phase to run"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="congress_corpus_data",
        help="Base directory for all data"
    )
    args = parser.parse_args()

    global DATA_DIR, LIST_DIR, HTML_DIR, SEGMENT_DIR, EXPORT_DIR
    DATA_DIR = Path(args.data_dir)
    LIST_DIR = DATA_DIR / "listings"
    HTML_DIR = DATA_DIR / "html"
    SEGMENT_DIR = DATA_DIR / "segments"
    EXPORT_DIR = DATA_DIR / "beir_export"

    if args.phase in ("list", "all"):
        logger.info("=== Phase 1: Listing hearings ===")
        asyncio.run(list_hearings())

    if args.phase in ("download", "all"):
        logger.info("=== Phase 2: Downloading hearings ===")
        asyncio.run(download_hearings())

    if args.phase in ("segment", "all"):
        logger.info("=== Phase 3: Segmenting into passages ===")
        segment_all_hearings()

    if args.phase in ("export", "all"):
        logger.info("=== Phase 4: Exporting to BEIR format ===")
        export_beir()


if __name__ == "__main__":
    main()
