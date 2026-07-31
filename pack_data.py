"""Pack tokenized examples into fixed-length sequences.

Packing here means: concatenate every document's ids end to end and cut the
stream at ``max_seq_length`` boundaries. Nothing is truncated (a long document
simply spans several sequences) and nothing is padded (only the very last
sequence may be short), so token utilisation is 100%.

Because the operation is a pure reshape of one contiguous id stream, it is done
in Arrow rather than in Python. ``pack_dataset_generator`` -- the original
row-by-row implementation -- is kept as the reference the fast path is tested
against (``tests/test_packing.py``): at 30B tokens, materialising every token as
a Python int would take the better part of a day, while the Arrow path slices
zero-copy and runs in seconds per shard.

    python pack_data.py --input-dir tokenized_data/train \
        --output-dir tokenized_data/packed_train_data_2048
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import datasets
import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

PACKED_FEATURES = datasets.Features(
    {"input_ids": datasets.List(datasets.Value("int32"))}
)

# Packed rows per Arrow chunk. Keeps each chunk's offset buffer inside int32
# (the width `List(int32)` uses) no matter how large the shard is, and bounds
# the working set while the table is assembled.
ROWS_PER_CHUNK = 100_000


class PackingStats:
    """Track statistics during packing."""

    def __init__(self):
        self.total_tokens = 0
        self.num_sequences = 0
        self.num_examples = 0


def pack_dataset_generator(
    dataset, max_seq_length: int, eos_token_id: int = 0, stats: PackingStats = None
) -> Iterator[dict]:
    """Reference implementation: yield packed sequences row by row.

    Semantically identical to ``pack_dataset`` but orders of magnitude slower,
    since every token crosses the Arrow/Python boundary. Used by the tests and
    for small datasets; the pipeline uses ``pack_dataset``.
    """
    current_buffer = []

    for example in tqdm(
        dataset, desc="Packing examples", unit="ex", total=len(dataset)
    ):
        tokens = list(example["input_ids"])

        if stats:
            stats.num_examples += 1
            stats.total_tokens += len(tokens)

        while tokens:
            space_left = max_seq_length - len(current_buffer)

            if len(tokens) <= space_left:
                current_buffer.extend(tokens)
                tokens = []
            else:
                current_buffer.extend(tokens[:space_left])
                tokens = tokens[space_left:]

                if stats:
                    stats.num_sequences += 1

                yield {"input_ids": current_buffer}
                current_buffer = []

    if current_buffer:
        if stats:
            stats.num_sequences += 1

        yield {"input_ids": current_buffer}


def pack_dataset(dataset: Dataset, max_seq_length: int) -> tuple[Dataset, dict]:
    """Pack `dataset` by reshaping its id stream in Arrow.

    ``flatten()`` on the ``input_ids`` list column hands back the underlying
    contiguous values buffer -- exactly the concatenation packing is defined as
    -- so the work reduces to handing Arrow a new set of list offsets over
    slices of that same buffer. No token is copied.
    """
    values = dataset.data.column("input_ids").combine_chunks().flatten()
    total_tokens = len(values)

    n_full, remainder = divmod(total_tokens, max_seq_length)
    row_lengths = [max_seq_length] * n_full + ([remainder] if remainder else [])

    chunks = []
    position = 0
    for start in range(0, len(row_lengths), ROWS_PER_CHUNK):
        block = row_lengths[start : start + ROWS_PER_CHUNK]
        n_bytes = sum(block)
        offsets = pa.array(np.cumsum([0] + block, dtype=np.int32))
        chunks.append(
            pa.ListArray.from_arrays(offsets, values.slice(position, n_bytes))
        )
        position += n_bytes

    column = pa.chunked_array(chunks, type=pa.list_(pa.int32()))
    packed = Dataset(
        pa.table({"input_ids": column}),
        info=datasets.DatasetInfo(features=PACKED_FEATURES),
    )

    stats = {
        "total_sequences": len(row_lengths),
        "total_tokens": total_tokens,
        "avg_tokens_per_sequence": total_tokens / max(1, len(row_lengths)),
        "avg_examples_per_sequence": len(dataset) / max(1, len(row_lengths)),
    }
    return packed, stats


def compute_stats_dict(stats: PackingStats, max_seq_length: int) -> dict:
    """Convert PackingStats to dictionary."""
    avg_examples_per_sequence = (
        stats.num_examples / stats.num_sequences if stats.num_sequences > 0 else 0
    )
    avg_tokens_per_sequence = (
        stats.total_tokens / stats.num_sequences if stats.num_sequences > 0 else 0
    )

    return {
        "total_sequences": stats.num_sequences,
        "total_tokens": stats.total_tokens,
        "avg_tokens_per_sequence": avg_tokens_per_sequence,
        "avg_examples_per_sequence": avg_examples_per_sequence,
    }


def pack_split(dataset, max_seq_length: int, eos_token_id: int = 0):
    """Pack a single split and report the statistics."""
    print(f"\n{'=' * 60}")
    print(f"Original examples: {len(dataset):,}")

    packed_dataset, stats = pack_dataset(dataset, max_seq_length)

    print("\nPacking Statistics")
    print(f"  Original examples: {len(dataset):,}")
    print(f"  Packed sequences: {stats['total_sequences']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Avg tokens per sequence: {stats['avg_tokens_per_sequence']:.1f}")
    print(f"  Avg examples per sequence: {stats['avg_examples_per_sequence']:.2f}")
    print("  No padding, no truncation - 100% token utilization")

    return packed_dataset, stats


def main():
    parser = argparse.ArgumentParser(
        description="Pack tokenized dataset into fixed-length sequences"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for packed sequences (default: 2048)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="tokenized_data/train",
        help="Input directory containing tokenized dataset (default: tokenized_data/train)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for packed dataset (default: tokenized_data/packed_train_data_{max_seq_length})",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Save a plain Dataset instead of a DatasetDict with a 'train' split.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"tokenized_data/packed_train_data_{args.max_seq_length}"

    print(f"\n{'=' * 60}")
    print("Dataset Packing Configuration")
    print(f"{'=' * 60}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    print(f"\nLoading tokenized dataset from {args.input_dir} ...")
    dataset = load_from_disk(args.input_dir)
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    packed_dataset, stats = pack_split(dataset, args.max_seq_length)
    all_stats = {"train": stats}

    print(f"\n{'=' * 60}")
    print("Saving packed dataset ...")
    print(f"{'=' * 60}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    to_save = packed_dataset if args.flat else DatasetDict({"train": packed_dataset})
    to_save.save_to_disk(args.output_dir, num_proc=16)
    print(f"Packed dataset saved to: {args.output_dir}")

    stats_path = output_path / "packing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "config": {
                    "max_seq_length": args.max_seq_length,
                    "input_dir": args.input_dir,
                },
                "statistics": all_stats,
            },
            f,
            indent=2,
        )
    print(f"Packing statistics saved to: {stats_path}")

    print(f"\n{'=' * 60}")
    print("Packing Complete!")
    print(f"{'=' * 60}")
    for split_name, stats in all_stats.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Sequences: {stats['total_sequences']:,}")
        print(f"  Avg tokens/seq: {stats['avg_tokens_per_sequence']:.1f}")
        print(f"  Avg examples/seq: {stats['avg_examples_per_sequence']:.2f}")


if __name__ == "__main__":
    main()
