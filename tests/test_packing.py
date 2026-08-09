"""The Arrow packing path must agree with the row-by-row reference exactly.

``pack_dataset`` is the implementation the pipeline uses because it slices Arrow
buffers instead of materialising 30B Python ints; ``pack_dataset_generator`` is
the original and is much easier to read. These tests are what licenses the swap.
"""

import datasets
import numpy as np
import pytest
from datasets import Dataset

from pretrain.data.packing import pack_dataset, pack_dataset_generator

FEATURES = datasets.Features({"input_ids": datasets.List(datasets.Value("int32"))})


def make_dataset(lengths) -> Dataset:
    rng = np.random.default_rng(0)
    rows = [rng.integers(0, 50258, size=int(n), dtype=np.int32).tolist() for n in lengths]
    return Dataset.from_dict({"input_ids": rows}, features=FEATURES)


@pytest.mark.parametrize(
    "lengths, max_seq_length",
    [
        # Ragged documents, both shorter and longer than the sequence length.
        (np.random.default_rng(1).integers(1, 400, size=300), 128),
        # Exact multiple: no short tail row.
        ([64, 64, 128], 64),
        # A single document spanning several sequences (no truncation).
        ([1000], 128),
        # Documents shorter than one sequence each.
        ([1, 2, 3, 4], 128),
        # One document, shorter than a sequence: a single short row.
        ([5], 128),
    ],
)
def test_matches_reference(lengths, max_seq_length):
    dataset = make_dataset(lengths)
    reference = [row["input_ids"] for row in pack_dataset_generator(dataset, max_seq_length)]
    packed, _ = pack_dataset(dataset, max_seq_length)

    assert packed["input_ids"] == reference


def test_invariants():
    dataset = make_dataset(np.random.default_rng(2).integers(1, 900, size=500))
    packed, stats = pack_dataset(dataset, 2048)

    lengths = [len(row) for row in packed["input_ids"]]
    # No padding and no truncation: every row is full except possibly the last.
    assert all(n == 2048 for n in lengths[:-1])
    assert 0 < lengths[-1] <= 2048
    # 100% utilisation: not one token gained or lost.
    assert sum(lengths) == sum(len(row) for row in dataset["input_ids"])
    assert stats["total_tokens"] == sum(lengths)
    assert stats["total_sequences"] == len(lengths)
    assert packed.features == FEATURES


def test_spans_several_arrow_chunks(monkeypatch):
    """The chunked assembly must not drop or reorder tokens at chunk borders."""
    monkeypatch.setattr("pretrain.data.packing.ROWS_PER_CHUNK", 3)
    dataset = make_dataset(np.random.default_rng(3).integers(1, 200, size=200))
    reference = [row["input_ids"] for row in pack_dataset_generator(dataset, 64)]
    packed, _ = pack_dataset(dataset, 64)

    assert packed["input_ids"] == reference
