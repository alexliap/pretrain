"""Shared fixtures for the resume-mechanism test suite.

Everything here is CPU-only and uses tiny synthetic data so the tests are fast and
deterministic — the resume logic under test lives in accelerate and the dataloader,
not in the model, so no real LLM/GPU/flash-attn is needed.
"""

import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset, DatasetDict


@pytest.fixture
def cpu_accelerator():
    """An Accelerator pinned to CPU (single process)."""
    return Accelerator(cpu=True)


@pytest.fixture
def packed_dataset(tmp_path, monkeypatch):
    """Create a deterministic on-disk packed dataset and chdir to its root.

    Row ``i`` has ``input_ids == [i] * max_seq_length``, so each batch can be
    identified by ``input_ids[:, 0]``. It is saved exactly where
    ``PretrainDataLoader.train_dataloader`` looks for it
    (``tokenized_data/packed_train_data_<L>``), relative to the monkeypatched cwd.
    """
    max_seq_length = 8
    num_rows = 24

    monkeypatch.chdir(tmp_path)
    dataset = Dataset.from_dict(
        {"input_ids": [[i] * max_seq_length for i in range(num_rows)]}
    )
    DatasetDict({"train": dataset}).save_to_disk(
        f"tokenized_data/packed_train_data_{max_seq_length}"
    )
    return {"max_seq_length": max_seq_length, "num_rows": num_rows}


@pytest.fixture
def tiny_model_optim():
    """Factory for a tiny model + AdamW, mirroring _init_optimizer_and_scheduler."""

    def _make(lr: float = 1e-4):
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        return model, optimizer

    return _make
