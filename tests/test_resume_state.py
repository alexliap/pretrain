"""Training counters must survive a checkpoint round-trip.

``step_in_epoch`` is what drives the dataloader batch-skip on resume, so it must
persist through the exact ``register_for_checkpointing`` + ``save_state`` /
``load_state`` cycle the resume path uses (task.py:489, task.py:504).
"""

from pretrain.checkpoint import TrainingState


def test_training_state_roundtrip():
    state = TrainingState()
    state.epoch = 3
    state.global_step = 123
    state.step_in_epoch = 45
    state.tokens_seen = 6789

    restored = TrainingState()
    restored.load_state_dict(state.state_dict())

    assert restored.state_dict() == state.state_dict()


def test_save_load_restores_counters(tmp_path, cpu_accelerator, tiny_model_optim):
    model, optimizer = tiny_model_optim()
    model, optimizer = cpu_accelerator.prepare(model, optimizer)

    state = TrainingState()
    cpu_accelerator.register_for_checkpointing(state)

    state.epoch = 2
    state.global_step = 100
    state.step_in_epoch = 50
    state.tokens_seen = 123_456

    state_dir = str(tmp_path / "state")
    cpu_accelerator.save_state(state_dir)

    # Wipe the in-memory counters, then prove load_state repopulates them from disk.
    state.epoch = 0
    state.global_step = 0
    state.step_in_epoch = 0
    state.tokens_seen = 0

    cpu_accelerator.load_state(state_dir)

    assert state.epoch == 2
    assert state.global_step == 100
    assert state.step_in_epoch == 50
    assert state.tokens_seen == 123_456
