"""Tokenize and pack the mixed shards into the training store.

One shard at a time: read its parquet, encode it, reshape the id stream into
``max_seq_length`` rows, save, free. Nothing intermediate is persisted -- the
tokenized-but-unpacked form only ever exists in memory -- so the disk holds the
mixed text (61 GB) and the packed ids (~120 GB) and nothing else. Doing the whole
30B corpus in one piece would instead need those plus two full Arrow caches.

    python prepare_shards.py                    # all shards, resumable
    python prepare_shards.py --shards 0 1 2     # a subset
    python prepare_shards.py --stats-only       # re-aggregate the JSON reports

Shards already carrying a ``shard_stats.json`` are skipped, so an interrupted run
resumes by re-running the same command.

Output layout::

    tokenized_data/
      test/                                  # held out, unpacked, variable length
      token_distribution.json                # read by task.py
      packed_train_data_2048/
        shard_00/ ... shard_29/              # concatenated by dataloader.py
        packing_stats.json
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

from pack_data import pack_dataset
from pretrain.data.tokenize import (
    count_tokens_per_source,
    load_tokenizer,
    save_token_distribution,
    tokenize_shard,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MIXED_DIR = Path("data/mixed_dataset")
OUTPUT_DIR = Path("tokenized_data")
TOKENIZER_PATH = "models/bilingual_el_en_50k"
MAX_SEQ_LENGTH = 2048
NUM_PROC = 48

# Rows held out for validation, taken from the tail of the last shard. They are
# left unpacked and variable-length because ``val_dataloader`` builds an
# attention mask from real sequence lengths -- packing them would make every
# validation sequence a splice of unrelated documents.
TEST_ROWS = 50_000


def shard_paths(mixed_dir: Path) -> list[Path]:
    paths = sorted(mixed_dir.glob("shard_*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No shard_*.parquet under {mixed_dir}. Run `python concat_data.py` first."
        )
    return paths


def _read_shard(path: Path, offset: int = 0, length: int | None = None) -> Dataset:
    """Load a mixed shard into memory as a ``(text, dataset)`` Dataset.

    Read through pyarrow rather than ``load_dataset`` so the shard never lands in
    the HF datasets cache: that cache would be a second 61 GB copy of the text,
    and it outlives ``cleanup_cache_files()``.
    """
    table = pq.read_table(path, columns=["text", "dataset"])
    if length is not None:
        table = table.slice(offset, length)
    elif offset:
        table = table.slice(offset)
    return Dataset(table)


def build_test_split(
    path: Path, tokenizer, output_dir: Path, test_rows: int, num_proc: int
) -> None:
    """Hold out the last `test_rows` documents of `path`, tokenized and unpacked."""
    test_dir = output_dir / "test"
    if (test_dir / "dataset_info.json").exists():
        logger.info("test split already present at %s, skipping", test_dir)
        return

    n_rows = pq.ParquetFile(path).metadata.num_rows
    dataset = _read_shard(path, offset=n_rows - test_rows, length=test_rows)
    tokenized = tokenize_shard(
        dataset, tokenizer, num_proc=num_proc, desc="Tokenizing test split"
    )
    n_tokens = sum(count_tokens_per_source(tokenized).values())
    tokenized = tokenized.remove_columns(["dataset", "num_tokens"])
    tokenized.save_to_disk(test_dir)
    logger.info(
        "test split: %d documents, %d tokens -> %s", len(tokenized), n_tokens, test_dir
    )


def prepare_shard(
    path: Path,
    index: int,
    tokenizer,
    packed_dir: Path,
    max_seq_length: int,
    num_proc: int,
    row_limit: int | None,
) -> dict:
    """Tokenize, pack and save one shard; return its stats."""
    out_dir = packed_dir / f"shard_{index:02d}"
    stats_path = out_dir / "shard_stats.json"
    if stats_path.exists():
        logger.info("shard %02d already packed, skipping", index)
        return json.loads(stats_path.read_text())

    dataset = _read_shard(path, length=row_limit)
    tokenized = tokenize_shard(
        dataset, tokenizer, num_proc=num_proc, desc=f"Tokenizing shard {index:02d}"
    )
    del dataset

    tokens_per_source = count_tokens_per_source(tokenized)
    packed, packing = pack_dataset(
        tokenized.remove_columns(["dataset", "num_tokens"]), max_seq_length
    )
    del tokenized

    packed.save_to_disk(out_dir, num_proc=16)
    del packed

    stats = {"tokens_per_source": tokens_per_source, "packing": packing}
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(
        "shard %02d: %d sequences, %d tokens -> %s",
        index,
        packing["total_sequences"],
        packing["total_tokens"],
        out_dir,
    )
    return stats


def aggregate(packed_dir: Path, output_dir: Path, max_seq_length: int) -> None:
    """Merge the per-shard reports into the corpus-level ones."""
    per_shard = {}
    for stats_path in sorted(packed_dir.glob("shard_*/shard_stats.json")):
        per_shard[stats_path.parent.name] = json.loads(stats_path.read_text())

    if not per_shard:
        logger.warning("No shard_stats.json under %s -- nothing to aggregate", packed_dir)
        return

    tokens_per_source: dict[str, int] = defaultdict(int)
    total_sequences = 0
    total_tokens = 0
    total_examples = 0.0
    for stats in per_shard.values():
        for source, n in stats["tokens_per_source"].items():
            tokens_per_source[source] += n
        total_sequences += stats["packing"]["total_sequences"]
        total_tokens += stats["packing"]["total_tokens"]
        total_examples += (
            stats["packing"]["avg_examples_per_sequence"]
            * stats["packing"]["total_sequences"]
        )

    save_token_distribution(dict(tokens_per_source), str(output_dir))

    (packed_dir / "packing_stats.json").write_text(
        json.dumps(
            {
                "config": {
                    "max_seq_length": max_seq_length,
                    "n_shards": len(per_shard),
                },
                "statistics": {
                    "train": {
                        "total_sequences": total_sequences,
                        "total_tokens": total_tokens,
                        "avg_tokens_per_sequence": total_tokens
                        / max(1, total_sequences),
                        "avg_examples_per_sequence": total_examples
                        / max(1, total_sequences),
                    }
                },
                "per_shard": {
                    name: stats["packing"] for name, stats in per_shard.items()
                },
            },
            indent=2,
        )
    )

    logger.info("")
    logger.info(
        "%-16s %14s %8s", "source", "tokens", "share",
    )
    for source, n in sorted(tokens_per_source.items(), key=lambda kv: -kv[1]):
        logger.info("%-16s %14d %7.1f%%", source, n, 100 * n / total_tokens)
    logger.info(
        "TOTAL %d shards, %d sequences, %.3fB tokens (%.1f tokens/seq)",
        len(per_shard),
        total_sequences,
        total_tokens / 1e9,
        total_tokens / max(1, total_sequences),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize and pack the mixed shards into the training store."
    )
    parser.add_argument("--mixed-dir", type=Path, default=MIXED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--num-proc", type=int, default=NUM_PROC)
    parser.add_argument("--test-rows", type=int, default=TEST_ROWS)
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=None,
        help="Shard indices to process (default: all).",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Rows per shard, for a quick end-to-end check.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Re-aggregate the per-shard reports without processing anything.",
    )
    args = parser.parse_args()

    packed_dir = args.output_dir / f"packed_train_data_{args.max_seq_length}"
    packed_dir.mkdir(parents=True, exist_ok=True)

    if args.stats_only:
        aggregate(packed_dir, args.output_dir, args.max_seq_length)
        return

    paths = shard_paths(args.mixed_dir)
    selected = (
        list(enumerate(paths))
        if args.shards is None
        else [(i, paths[i]) for i in args.shards]
    )
    logger.info(
        "Preparing %d/%d shards from %s at max_seq_length=%d",
        len(selected),
        len(paths),
        args.mixed_dir,
        args.max_seq_length,
    )

    tokenizer = load_tokenizer(args.tokenizer)

    # The held-out documents come off the tail of the last shard, and that same
    # tail is excluded from training below.
    last_index = len(paths) - 1
    build_test_split(
        paths[last_index], tokenizer, args.output_dir, args.test_rows, args.num_proc
    )

    for index, path in selected:
        row_limit = args.row_limit
        if index == last_index and row_limit is None:
            row_limit = pq.ParquetFile(path).metadata.num_rows - args.test_rows
        prepare_shard(
            path,
            index,
            tokenizer,
            packed_dir,
            args.max_seq_length,
            args.num_proc,
            row_limit,
        )

    aggregate(packed_dir, args.output_dir, args.max_seq_length)


if __name__ == "__main__":
    main()
