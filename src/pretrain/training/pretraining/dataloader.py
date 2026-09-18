import logging
from functools import partial
from pathlib import Path

import torch
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def collate_fn(
    batch,
    max_seq_length: int,
    return_attention_mask: bool = False,
    pad_to_max: bool = False,
):
    """Pad input_ids so they can be stacked into one tensor.

    When return_attention_mask is True (e.g. for non-packed validation data),
    an attention_mask is built from the real sequence lengths so the model does
    not attend to padding.

    When pad_to_max is True, every batch is padded to a constant width
    (max_seq_length) instead of the per-batch longest sequence. This keeps the
    batch shape invariant across batches, avoiding torch.compile recompiles
    that variable sequence lengths would otherwise trigger.
    """
    input_ids = [item["input_ids"][:max_seq_length] for item in batch]
    lengths = torch.tensor([len(x) for x in input_ids])

    if pad_to_max:
        padded_input_ids = torch.zeros(
            (len(input_ids), max_seq_length), dtype=input_ids[0].dtype
        )
        for i, ids in enumerate(input_ids):
            padded_input_ids[i, : ids.size(0)] = ids
    else:
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    # The store holds int32 ids to halve its size on disk. nn.Embedding accepts
    # int32, but task.py derives `labels` from these ids and cross-entropy
    # requires int64 targets, so widen here - once per batch, on a small tensor.
    out = {"input_ids": padded_input_ids.long()}
    if return_attention_mask:
        positions = torch.arange(padded_input_ids.size(1))
        out["attention_mask"] = (positions[None, :] < lengths[:, None]).long()
    return out


class PretrainDataLoader:
    def __init__(
        self,
        train_batch_size=16,
        val_batch_size=4,
        num_workers=4,
        max_seq_length=2048,
        use_packed_data=True,
    ):
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.use_packed_data = use_packed_data

    def _load_train(self):
        """Load the training store, sharded or not.

        ``prepare_shards.py`` writes one directory per shard instead of one
        monolithic dataset: consolidating 30 shards into a single
        ``save_to_disk`` would copy ~120 GB for no benefit.
        ``concatenate_datasets`` joins them in sorted shard order, and that order
        is what makes resume-by-sample-index correct - the training loader runs
        with ``shuffle=False`` so the on-disk order is the training order
        (tests/test_resume_dataloader.py::test_deterministic_order).
        """
        if self.use_packed_data:
            data_path = Path(f"tokenized_data/packed_train_data_{self.max_seq_length}")
        else:
            data_path = Path("tokenized_data/train")

        # `save_to_disk` writes the arrow files first and `state.json` last, so
        # its presence is what distinguishes a finished shard from one still being
        # written. Filtering on it lets a run start against a store that
        # prepare_shards.py is still filling - useful for test runs - instead of
        # dying on the half-written directory.
        all_dirs = sorted(data_path.glob("shard_*"))
        shard_dirs = [d for d in all_dirs if (d / "state.json").exists()]

        if shard_dirs:
            skipped = len(all_dirs) - len(shard_dirs)
            if skipped:
                logger.warning(
                    "%s: %d of %d shard directories are incomplete and were "
                    "skipped - this run sees a partial corpus.",
                    data_path,
                    skipped,
                    len(all_dirs),
                )
            train = concatenate_datasets(
                [load_from_disk(str(shard_dir)) for shard_dir in shard_dirs]
            )
            # Logged unconditionally: a store that is short a few shards is
            # otherwise indistinguishable from a complete one at training time.
            logger.info(
                "Loaded %d shards from %s: %d sequences (~%.2fB tokens)",
                len(shard_dirs),
                data_path,
                len(train),
                len(train) * self.max_seq_length / 1e9,
            )
            return train

        dataset = load_from_disk(str(data_path))
        # A DatasetDict from the single-shot tokenize path, a Dataset otherwise.
        return dataset["train"] if isinstance(dataset, DatasetDict) else dataset

    def train_dataloader(self):
        train = self._load_train()

        # Remove attention_mask if it exists (may not exist in packed data)
        if "attention_mask" in train.column_names:
            train = train.remove_columns(["attention_mask"])

        train.set_format(type="torch", columns=["input_ids"])

        return DataLoader(
            train,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=True,
            # Packed training data has no padding, so no attention mask is needed.
            # Passing one would push the model onto FlashAttention's variable-length
            # path (the .item() graph break) and trigger torch.compile recompiles.
            collate_fn=partial(collate_fn, max_seq_length=self.max_seq_length),
        )

    def val_dataloader(self, size: int | None = int(5e2)):
        data_path = "tokenized_data/test"

        val = load_from_disk(data_path)

        # Remove attention_mask if it exists (may not exist in packed data)
        if "attention_mask" in val.column_names:
            val = val.remove_columns(["attention_mask"])

        # size=None means "use the entire held-out split" instead of a fixed sample.
        if size is not None:
            val = val.select(range(size))
        val.set_format(type="torch", columns=["input_ids"])

        return DataLoader(
            val,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=partial(
                collate_fn,
                max_seq_length=self.max_seq_length,
                return_attention_mask=True,
            ),
        )
