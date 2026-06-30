import glob
import json
import os

import polars as pl

SEED = 0

# Number of rows to keep per dataset. Keys must match the dataset directory
# names under data/ (e.g. "finewiki_el"). Use None to keep all rows.
DATASET_WEIGHTS = {
    "finewiki_el": 200_000,
    "fineweb_hq_el": 1_000_000,
    "math": 300_000,
    "table": 300_000,
    "faq": 300_000,
    "tutorial": 300_000,
    "enwiki": 400_000,
}


if __name__ == "__main__":
    # Map each dataset (directory under data/) to the files it contains.
    files_by_dataset: dict[str, list[str]] = {}
    for path in glob.glob("data/*/*.jsonl") + glob.glob("data/*/*.parquet"):
        if "finepdfs" in path:
            continue
        # data/finewiki_el/data.parquet -> "finewiki_el"
        dataset = os.path.basename(os.path.dirname(path))
        files_by_dataset.setdefault(dataset, []).append(path)

    # Fail early, before any filtering, if a weighted dataset is missing on disk.
    missing = [name for name in DATASET_WEIGHTS if name not in files_by_dataset]
    if missing:
        raise ValueError(
            f"Datasets defined in DATASET_WEIGHTS but not found under data/: {missing}. "
            f"Available datasets: {sorted(files_by_dataset)}"
        )

    frames = []
    rows_per_dataset: dict[str, int] = {}
    for dataset, n_rows in DATASET_WEIGHTS.items():
        parts = [
            pl.scan_ndjson(p) if p.endswith(".jsonl") else pl.scan_parquet(p)
            for p in files_by_dataset[dataset]
        ]
        frame = pl.concat(parts).select(["text"])

        if n_rows is not None:
            print(f"Sampling {n_rows} rows from {dataset} with seed {SEED} ...")
            frame = frame.collect().sample(n_rows, seed=SEED)
        else:
            frame = frame.collect()

        frame = frame.with_columns(pl.lit(dataset).alias("dataset"))
        rows_per_dataset[dataset] = frame.height
        frames.append(frame)

    total_rows = sum(rows_per_dataset.values())
    distribution = {
        "seed": SEED,
        "weights": DATASET_WEIGHTS,
        "total_rows": total_rows,
        "proportions": {k: round(v / total_rows, 3) for k, v in rows_per_dataset.items()},
    }

    os.makedirs("data/mixed_dataset", exist_ok=True)
    pl.concat(frames).write_parquet("data/mixed_dataset/data.parquet")
    with open("data/mixed_dataset/data_distribution.json", "w") as f:
        json.dump(distribution, f, indent=2)
