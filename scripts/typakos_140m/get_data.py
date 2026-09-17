import logging
from collections import defaultdict
from pathlib import Path

import polars as pl
from huggingface_hub import HfFileSystem, hf_hub_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Get direct HTTP URLs for the parquet files
    repo_id = "alexliap/greek-synth-v1"
    repo_type = "dataset"

    fs = HfFileSystem()
    file_paths = [
        p.removeprefix(f"datasets/{repo_id}/")
        for p in fs.glob(f"datasets/{repo_id}/**/*.parquet")
        # if "/enwiki/" in p
    ]

    urls_by_dataset: dict[str, list[str]] = defaultdict(list)
    for path in file_paths:
        dataset = path.split("/", 1)[0]
        urls_by_dataset[dataset].append(
            hf_hub_url(repo_id=repo_id, filename=path, repo_type=repo_type)
        )

    for dataset, urls in urls_by_dataset.items():
        out_dir = Path("data") / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Reading {len(urls)} files for {dataset}")
        pl.scan_parquet(urls).sink_parquet(out_dir / "train_data.parquet")
