"""The dataloader must skip previously-seen batches on resume.

These exercise the real ``PretrainDataLoader.train_dataloader`` and the real
``accelerator.skip_first_batches`` — the exact pieces the resume path wires together
in ``PretrainTask.train`` (task.py:594-608).
"""

import pytest

from pretrain.dataloader import PretrainDataLoader

BATCH_SIZE = 4


def _batch_row_ids(loader):
    """Row index carried by each example, grouped per batch (input_ids[:, 0])."""
    return [batch["input_ids"][:, 0].tolist() for batch in loader]


def _make_train_loader(params, batch_size=BATCH_SIZE):
    factory = PretrainDataLoader(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=2,
        max_seq_length=params["max_seq_length"],
        use_packed_data=True,
    )
    return factory.train_dataloader()


@pytest.mark.parametrize("skip", [1, 2, 3])
def test_skip_first_batches_yields_unseen_remainder(
    packed_dataset, cpu_accelerator, skip
):
    """After skipping N batches, the loader resumes exactly at batch N — no sample
    seen twice, none dropped."""
    loader = cpu_accelerator.prepare(_make_train_loader(packed_dataset))

    reference = _batch_row_ids(loader)

    skipped = cpu_accelerator.skip_first_batches(loader, skip)
    resumed = _batch_row_ids(skipped)

    # The resumed stream is precisely the tail the original run had not consumed.
    assert resumed == reference[skip:]

    # Consumed-before + resumed-after together cover every row exactly once.
    consumed = [row for batch in reference[:skip] for row in batch]
    remaining = [row for batch in resumed for row in batch]
    assert sorted(consumed + remaining) == list(range(packed_dataset["num_rows"]))


def test_skip_zero_is_identity(packed_dataset, cpu_accelerator):
    """skip=0 (the not-resuming branch) yields the full, unmodified stream."""
    loader = cpu_accelerator.prepare(_make_train_loader(packed_dataset))

    reference = _batch_row_ids(loader)
    identity = _batch_row_ids(cpu_accelerator.skip_first_batches(loader, 0))

    assert identity == reference


def test_deterministic_order(packed_dataset, cpu_accelerator):
    """Resume correctness relies on shuffle=False giving a stable order."""
    loader = cpu_accelerator.prepare(_make_train_loader(packed_dataset))

    first = _batch_row_ids(loader)
    second = _batch_row_ids(loader)

    assert first == second
    # Fixed-size batches in ascending row order.
    assert first[0] == [0, 1, 2, 3]
