"""Pack tokenized examples into fixed-length sequences.

Packing here means: concatenate every document's ids end to end and cut the
stream at ``max_seq_length`` boundaries. Nothing is truncated (a long document
simply spans several sequences) and nothing is padded (only the very last
sequence may be short), so token utilisation is 100%.

Because the operation is a pure reshape of one contiguous id stream, it is done
in Arrow rather than in Python. ``pack_dataset_generator`` - the original
row-by-row implementation - is kept as the reference the fast path is tested
against (``tests/test_packing.py``): at 30B tokens, materialising every token as
a Python int would take the better part of a day, while the Arrow path slices
zero-copy and runs in seconds per shard.

``pack_data.py`` at the repo root is the CLI over this module.
"""

from collections.abc import Iterator

import datasets
import numpy as np
import pyarrow as pa
from datasets import Dataset
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
    contiguous values buffer - exactly the concatenation packing is defined as
    - so the work reduces to handing Arrow a new set of list offsets over
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
