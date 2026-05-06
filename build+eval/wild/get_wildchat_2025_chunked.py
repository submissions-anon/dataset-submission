import duckdb
import json
import os
import gzip
from pathlib import Path
from tqdm import tqdm

OUT_DIR = Path("wildchat_2025_chunks")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MONTHS = [
    ("2025-01-01", "2025-02-01", "2025_01"),
    ("2025-02-01", "2025-03-01", "2025_02"),
    ("2025-03-01", "2025-04-01", "2025_03"),
    ("2025-04-01", "2025-05-01", "2025_04"),
    ("2025-05-01", "2025-06-01", "2025_05"),
    ("2025-06-01", "2025-07-01", "2025_06"),
    ("2025-07-01", "2025-08-01", "2025_07"),
    ("2025-08-01", "2025-09-01", "2025_08"),
    ("2025-09-01", "2025-10-01", "2025_09"),
    ("2025-10-01", "2025-11-01", "2025_10"),
    ("2025-11-01", "2025-12-01", "2025_11"),
    ("2025-12-01", "2026-01-01", "2025_12"),
]

def main():
    OUT_DIR.mkdir(exist_ok=True)

    con = duckdb.connect()
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
        print("No HF_TOKEN set — proceeding unauthenticated.")

    source = "hf://datasets/allenai/WildChat-4.8M/data/train-*.parquet"

    for start, end, tag in MONTHS:
        out_path = OUT_DIR / f"wildchat_{tag}_english.jsonl.gz"
        tmp_path = OUT_DIR / f"wildchat_{tag}_english.jsonl.gz.partial"

        if out_path.exists():
            print(f"[{tag}] already complete, skipping.")
            continue

        if tmp_path.exists():
            tmp_path.unlink()

        print(f"\n[{tag}] Counting rows...")
        count = con.sql(f"""
            SELECT COUNT(*)
            FROM read_parquet('{source}')
            WHERE timestamp >= '{start}'
              AND timestamp < '{end}'
              AND language = 'English'
        """).fetchone()[0]

        print(f"[{tag}] Found {count:,} conversations.")

        result = con.sql(f"""
            SELECT
                conversation_hash AS id,
                conversation,
                turn,
                timestamp
            FROM read_parquet('{source}')
            WHERE timestamp >= '{start}'
              AND timestamp < '{end}'
              AND language = 'English'
        """)

        written = 0

        with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
            with tqdm(total=count, desc=f"[{tag}]", unit="conv") as pbar:
                while True:
                    batch = result.fetchmany(250)
                    if not batch:
                        break

                    for conv_id, conversation, turn, timestamp in batch:
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
                            "timestamp": str(timestamp),
                            "conversation": clean_turns,
                            "turn": turn,
                        }

                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                        pbar.update(1)

        tmp_path.rename(out_path)
        print(f"[{tag}] Done. Wrote {written:,} rows to {out_path}")

if __name__ == "__main__":
    main()
