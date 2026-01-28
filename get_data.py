import logging

import polars as pl
from huggingface_hub import hf_hub_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Get direct HTTP URLs for the parquet files
    repo_id = "alexliap/bilingual-gr-en-text"
    repo_type = "dataset"

    # List a few files (you can expand this or use HfFileSystem to get all files)
    file_paths = [
        "el/train_data_el.parquet",
    ]

    # Get HTTP URLs
    urls = [
        hf_hub_url(repo_id=repo_id, filename=path, repo_type=repo_type)
        for path in file_paths
    ]

    print(f"Reading {len(urls)} files")

    # Scan parquet files from HTTP URLs
    pl.scan_parquet(urls).drop(["id", "metadata"]).sink_ndjson("data/train_data.jsonl")
