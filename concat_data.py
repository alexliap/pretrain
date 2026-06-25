import glob
import os

import polars as pl

if __name__ == "__main__":
    jsonl_files = []
    for path in glob.glob("data/*/*.jsonl", recursive=True):
        if "finepdfs" in path:
            continue
        jsonl_files.append(path)

    os.makedirs("data/concat_dataset", exist_ok=True)
    pl.scan_ndjson(jsonl_files).select(["text"]).sink_parquet(
        "data/concat_dataset/data.parquet"
    )
