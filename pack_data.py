#!/usr/bin/env python3
"""
Dataset Packing Script

Packs tokenized examples into fixed-length sequences to maximize context window
utilization during language model pretraining.

Usage:
    python pack_data.py --max_seq_length 2048 --input_dir tokenized_data/train_data --output_dir tokenized_data/packed_train_data_2048
"""

import argparse
import json
from pathlib import Path
from typing import Iterator

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm


class PackingStats:
    """Track statistics during packing."""

    def __init__(self):
        self.total_tokens = 0
        self.num_sequences = 0
        self.num_examples = 0


def pack_dataset_generator(
    dataset, max_seq_length: int, eos_token_id: int = 0, stats: PackingStats = None
) -> Iterator[dict]:
    """
    Generator that yields packed sequences from a dataset with 100% token utilization.

    This implementation:
    - Never truncates examples (splits long examples across multiple sequences)
    - No padding (last sequence can be shorter than max_seq_length)
    - Achieves 100% packing efficiency (zero waste)

    Args:
        dataset: HuggingFace dataset to pack
        max_seq_length: Target sequence length for packed sequences
        eos_token_id: Unused (kept for API compatibility)
        stats: Optional PackingStats object to track statistics during packing

    Yields:
        Packed sequences with 'input_ids' key (all max_seq_length except possibly the last one)
    """
    current_buffer = []

    for example in tqdm(
        dataset, desc="Packing examples", unit="ex", total=len(dataset)
    ):
        tokens = list(example["input_ids"])  # Convert to list for easier manipulation

        if stats:
            stats.num_examples += 1
            stats.total_tokens += len(tokens)

        # Process all tokens from this example
        while tokens:
            space_left = max_seq_length - len(current_buffer)

            if len(tokens) <= space_left:
                # All remaining tokens fit in current buffer
                current_buffer.extend(tokens)
                tokens = []
            else:
                # Fill current buffer to max_seq_length and yield it
                current_buffer.extend(tokens[:space_left])
                tokens = tokens[space_left:]  # Keep remaining tokens for next sequence

                if stats:
                    stats.num_sequences += 1

                yield {"input_ids": current_buffer}
                current_buffer = []

    # Handle final buffer (no padding - keep as variable length)
    if current_buffer:
        if stats:
            stats.num_sequences += 1

        yield {"input_ids": current_buffer}


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


def pack_split(
    dataset, max_seq_length: int, eos_token_id: int
) -> tuple[Dataset, dict]:
    """Pack a single split (train or test) of the dataset using dataset methods."""
    print(f"\n{'=' * 60}")
    print(f"Original examples: {len(dataset):,}")

    # Track statistics during packing
    packing_stats = PackingStats()

    # Use generator to create packed dataset efficiently with statistics tracking
    packed_dataset = Dataset.from_generator(
        lambda: pack_dataset_generator(
            dataset, max_seq_length, eos_token_id, packing_stats
        ),
    )

    # Convert statistics to dictionary
    stats = compute_stats_dict(packing_stats, max_seq_length)

    # Print statistics
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
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for packed sequences (default: 2048)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="tokenized_data/train_data",
        help="Input directory containing tokenized dataset (default: tokenized_data/train_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for packed dataset (default: tokenized_data/packed_train_data_{max_seq_length})",
    )
    parser.add_argument(
        "--eos_token_id",
        type=int,
        default=0,
        help="EOS/PAD token ID for padding (default: 0)",
    )

    args = parser.parse_args()

    # Set default output_dir if not provided
    if args.output_dir is None:
        args.output_dir = f"tokenized_data/packed_train_data_{args.max_seq_length}"

    print(f"\n{'=' * 60}")
    print("Dataset Packing Configuration")
    print(f"{'=' * 60}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"EOS/PAD token ID: {args.eos_token_id}")

    # Load tokenized dataset
    print(f"\nLoading tokenized dataset from {args.input_dir} ...")
    dataset = load_from_disk(args.input_dir)

    # Pack the (single-split) dataset
    packed_dataset, stats = pack_split(
        dataset, args.max_seq_length, args.eos_token_id
    )
    all_stats = {"train": stats}

    # Save packed dataset
    print(f"\n{'=' * 60}")
    print("Saving packed dataset ...")
    print(f"{'=' * 60}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as DatasetDict
    packed_dataset_dict = DatasetDict({"train": packed_dataset})
    packed_dataset_dict.save_to_disk(args.output_dir, num_proc=16)
    print(f"Packed dataset saved to: {args.output_dir}")

    # Save statistics to JSON
    stats_path = output_path / "packing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "config": {
                    "max_seq_length": args.max_seq_length,
                    "eos_token_id": args.eos_token_id,
                    "input_dir": args.input_dir,
                },
                "statistics": all_stats,
            },
            f,
            indent=2,
        )
    print(f"Packing statistics saved to: {stats_path}")

    # Print final summary
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
