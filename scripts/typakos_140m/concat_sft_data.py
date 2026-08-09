"""Concatenate the per-source SFT pools into one shuffled train/validation split.

Reads every source directory prepare_sft_data.py wrote under `data/sft_raw/`,
concatenates them (they all share the same messages/num_tokens/num_turns/source
schema), shuffles the pool, then splits off a validation fraction from the
back of that shuffled order. Output is a single `datasets.DatasetDict` with
"train"/"validation" splits, saved via `save_to_disk`, the format
`pretrain.sft.task.SFTTask._load_split` expects when `dataset_id` points at a
local directory.

    python scripts/typakos_140m/concat_sft_data.py
    python scripts/typakos_140m/concat_sft_data.py --val-fraction 0.05

Run from the repo root.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk
from prepare_sft_data import SOURCES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 0
VAL_FRACTION = 0.1
INPUT_DIR = Path("data/sft_raw")
OUTPUT_DIR = Path("data/sft_pool")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate the per-source SFT pools into one shuffled "
        "train/validation dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    sources = [load_from_disk(str(args.input_dir / name)) for name in sorted(SOURCES)]
    pool = concatenate_datasets(sources)
    logger.info("Concatenated %d sources -> %d rows", len(sources), len(pool))

    pool = pool.shuffle(seed=args.seed)
    logger.info("Shuffled with seed=%d", args.seed)

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

    for name in ("train", "validation"):
        counts = Counter(split[name]["source"])
        logger.info("%s per-source: %s", name, dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
