"""
collect_tweets.py
-----------------
Collect tweets about the Iran-US-Israel airstrikes (started Feb 27, 2026)
using the X API v2 recent search endpoint.

Requirements:
    pip install requests

Usage:
    export BEARER_TOKEN="your_bearer_token_here"
    python collect_tweets.py
"""

import os
import time
import json
import requests
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────

BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
OUTPUT_FILE  = "iran_israel_us_tweets_v2.jsonl"
MAX_TWEETS   = 100000          # how many we want total
RESULTS_PER_PAGE = 100       # max allowed by the API (10–100)

# Start time: just before the airstrikes began
START_TIME = "2026-02-27T00:00:00Z"

# Search query — OR across several relevant keyword clusters
# Exclude retweets to keep originals only (is:retweet removes RTs)
QUERY = (
    "(Iran airstrike OR Iran attack OR Iran strike OR Iran bomb "
    "OR Israel Iran OR US Iran OR Iran Israel US "
    "OR Tehran strike OR Persian Gulf attack "
    "OR IDF Iran OR IRGC strike) "
    "lang:en -is:retweet"
)

# Fields to request on each tweet
TWEET_FIELDS = ",".join([
    "id",
    "text",
    "created_at",
    "author_id",
    "public_metrics",    # likes, retweets, replies, quotes
    "lang",
    "entities",          # hashtags, urls, mentions
    "context_annotations",  # X's own topic labels (useful metadata)
])

# ── API call ──────────────────────────────────────────────────────────────────

def search_recent(query, start_time, next_token=None, max_results=100):
    url = "https://api.x.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query":        query,
        "start_time":   start_time,
        "max_results":  max_results,
        "tweet.fields": TWEET_FIELDS,
    }
    if next_token:
        params["next_token"] = next_token

    resp = requests.get(url, headers=headers, params=params)

    # Respect rate limits
    if resp.status_code == 429:
        reset_at = int(resp.headers.get("x-rate-limit-reset", time.time() + 60))
        wait = max(reset_at - int(time.time()), 1)
        print(f"  Rate limited. Sleeping {wait}s...")
        time.sleep(wait)
        return search_recent(query, start_time, next_token, max_results)

    resp.raise_for_status()
    return resp.json()

# ── Main collection loop ──────────────────────────────────────────────────────

def main():
    if not BEARER_TOKEN:
        raise ValueError("Set your BEARER_TOKEN environment variable first.")

    print(f"Collecting up to {MAX_TWEETS} tweets...")
    print(f"Query: {QUERY}\n")

    collected = []
    next_token = None

    with open(OUTPUT_FILE, "w") as f:
        while len(collected) < MAX_TWEETS:
            batch_size = max(10, min(RESULTS_PER_PAGE, MAX_TWEETS - len(collected)))
            data = search_recent(QUERY, START_TIME, next_token, batch_size)

            tweets = data.get("data", [])
            meta   = data.get("meta", {})

            if not tweets:
                print("No more tweets found.")
                break

            for tweet in tweets:
                f.write(json.dumps(tweet) + "\n")

            collected.extend(tweets)
            print(f"  Collected {len(collected)} tweets so far...")

            next_token = meta.get("next_token")
            if not next_token:
                print("Reached end of results.")
                break

            # Small sleep to be polite to the API
            time.sleep(1)

    print(f"\nDone. {len(collected)} tweets saved to {OUTPUT_FILE}")
    print("Each line is a JSON object (jsonl format).")

if __name__ == "__main__":
    main()
