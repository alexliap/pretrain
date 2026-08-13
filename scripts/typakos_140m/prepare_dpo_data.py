"""Download and prepare the DPO preference dataset for typakos_140m.

Pulls the Greek and English configs of
openeurollm/Dolci-Instruct-DPO-translated (laid out on the Hub as
`el/shard*.parquet` / `en/shard*.parquet`), concatenates them, shuffles, splits
off a validation fraction, and saves the result as a single
`datasets.DatasetDict` via `save_to_disk` -- the format
`pretrain.dpo.task.DPOTask._load_split` expects when `dataset_id` points at a
local directory. Rows already match the "conversational" DPO schema
(`prompt`/`chosen`/`rejected`, each a list of chat messages), so no reshaping
is needed beyond dropping the `id` column and concatenating languages.

    python scripts/typakos_140m/prepare_dpo_data.py
    python scripts/typakos_140m/prepare_dpo_data.py --languages el en --val-fraction 0.02

Run from the repo root.
"""

import argparse
import logging
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets

from pretrain.data.download import download_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ID = "openeurollm/Dolci-Instruct-DPO-translated"
LANGUAGES = ["el", "en"]
KEEP_COLUMNS = ["prompt", "chosen", "rejected"]
SEED = 0
VAL_FRACTION = 0.1
RAW_DIR = Path("data/raw_dpo")
OUTPUT_DIR = Path("data/typakos_dpo_dataset")


def load_language(lang: str, raw_dir: Path) -> Dataset:
    local_dir = download_dataset(
        REPO_ID, raw_dir / lang, allow_patterns=[f"{lang}/*.parquet"]
    )
    files = sorted(local_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {local_dir}")
    table = pq.read_table(files, columns=KEEP_COLUMNS)
    return Dataset(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the el/en Dolci-Instruct-DPO-translated pool."
    )
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    per_language = []
    for lang in args.languages:
        dataset = load_language(lang, args.raw_dir)
        logger.info("%s: %d rows", lang, len(dataset))
        per_language.append(dataset)

    pool = concatenate_datasets(per_language)
    pool = pool.shuffle(seed=args.seed)
    logger.info("Concatenated %d languages -> %d rows", len(per_language), len(pool))

    # Already shuffled above, so a plain (non-shuffling) split off the back of
    # that order is itself a random validation slice.
    split = pool.train_test_split(test_size=args.val_fraction, shuffle=False)
    split["validation"] = split.pop("test")

    split.save_to_disk(str(args.output_dir))
    logger.info(
        "Saved to %s: train=%d, validation=%d",
        args.output_dir,
        len(split["train"]),
        len(split["validation"]),
    )


if __name__ == "__main__":
    main()
