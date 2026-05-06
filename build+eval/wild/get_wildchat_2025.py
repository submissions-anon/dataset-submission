import duckdb
import json
import os
from pathlib import Path
from tqdm import tqdm

OUTPUT_FILE = "wildchat_2025_english.jsonl"
HF_TOKEN    = os.environ.get("HF_TOKEN", "")

def main():
    con = duckdb.connect()

    # hf:// support comes from httpfs
    con.install_extension("httpfs")
    con.load_extension("httpfs")

    if HF_TOKEN:
        con.sql(f"""
            CREATE SECRET hf_token (
                TYPE HUGGINGFACE,
                TOKEN '{HF_TOKEN}'
            )
        """)
        print("Authenticated with HuggingFace token.")
    else:
        print("No HF_TOKEN set — proceeding unauthenticated (may hit rate limits).")

    print("Counting 2025 English conversations...")
    count = con.sql("""
        SELECT COUNT(*)
        FROM read_parquet('hf://datasets/allenai/WildChat-4.8M/data/train-*.parquet')
        WHERE timestamp >= '2025-01-01'
          AND language = 'English'
    """).fetchone()[0]
    print(f"  Found {count:,} conversations to download.")

    print(f"Streaming to {OUTPUT_FILE}...")
    out_path = Path(OUTPUT_FILE)

    already_done = 0
    if out_path.exists():
        with open(out_path) as f:
            already_done = sum(1 for _ in f)
        print(f"  Resuming from {already_done:,} already written.")

    result = con.sql("""
        SELECT
            row_number() OVER () AS rn,
            conversation_hash AS id,
            conversation,
            turn
        FROM read_parquet('hf://datasets/allenai/WildChat-4.8M/data/train-*.parquet')
        WHERE timestamp >= '2025-01-01'
          AND language = 'English'
    """)

    written = 0
    skipped = 0

    with open(out_path, "a") as f:
        with tqdm(total=count, initial=already_done, desc="Downloading", unit="conv") as pbar:
            while True:
                batch = result.fetchmany(1000)
                if not batch:
                    break
                for row in batch:
                    rn, conv_id, conversation, turn = row
                    if skipped < already_done:
                        skipped += 1
                        continue

                    clean_turns = []
                    if conversation:
                        for turn_obj in conversation:
                            if isinstance(turn_obj, dict):
                                clean_turns.append({
                                    "role": turn_obj.get("role", ""),
                                    "content": turn_obj.get("content", ""),
                                })

                    record = {
                        "id": str(conv_id),
                        "conversation": clean_turns,
                        "turn": turn,
                    }
                    f.write(json.dumps(record) + "\n")
                    written += 1
                    pbar.update(1)

    print(f"\nDone. Wrote {written:,} conversations to {OUTPUT_FILE}")
    print(f"File size: {out_path.stat().st_size / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
