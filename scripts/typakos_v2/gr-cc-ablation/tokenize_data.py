import argparse
import logging
from pathlib import Path

from pretrain.data.common import parquet_paths, read_parquet
from pretrain.data.tokenize import (
    load_tokenizer,
    tokenize_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

TOKENIZER_PATH = "alexliap/typakos_v2_tokenizer"
NUM_PROC = 8

crawls = [
    "CC-MAIN-2024-22",
    "CC-MAIN-2024-26",
    "CC-MAIN-2024-30",
    "CC-MAIN-2024-33",
    "CC-MAIN-2024-38",
    "CC-MAIN-2024-42",
    "CC-MAIN-2024-46",
    "CC-MAIN-2024-51",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize and pack the mixed shards into the training store."
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--num-proc", type=int, default=NUM_PROC)

    args = parser.parse_args()

    tokenizer = load_tokenizer(TOKENIZER_PATH)

    for crawl in crawls:
        paths = parquet_paths(Path(f"data/raw/{args.dataset_path}/{crawl}"))
        dataset = read_parquet(paths, columns=["text", "crawl"]).rename_column(
            "crawl", "dataset"
        )
        tokenized = tokenize_dataset(dataset, tokenizer, num_proc=args.num_proc)

        tokenized.save_to_disk(
            f"data/tokenized/{args.dataset_path}/{crawl}",
            num_proc=args.num_proc,
            max_shard_size="400MB",
        )
        del tokenized


if __name__ == "__main__":
    main()
