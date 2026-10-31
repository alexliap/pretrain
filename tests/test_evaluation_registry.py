"""EvaluationRunner must build one task per registered TASK_REGISTRY entry,
with no network access needed at construction time."""

from unittest.mock import MagicMock, patch

import torch.nn as nn

from pretrain.config import EvaluationConfig, EvaluationTaskConfig
from pretrain.evaluation import TASK_REGISTRY, EvaluationRunner, EvaluationTask


class _RecordingTask(EvaluationTask):
    """Serves an ordered dataset and records which rows each run scored."""

    name = "recording"

    def __init__(self, num_samples):
        super().__init__(
            EvaluationTaskConfig(enabled=True, num_samples=num_samples),
            tokenizer=MagicMock(),
            accelerator=MagicMock(),
        )
        self.seen = []

    def load_data(self):
        return list(range(1000))

    def evaluate_batch(self, model, examples):
        self.seen.append(examples)
        return {"acc": 0.0}


def test_registry_contains_expected_tasks():
    assert {"mmlu", "hellaswag", "piqa"}.issubset(TASK_REGISTRY.keys())


def test_registry_contains_greek_tasks():
    assert {
        "mmlu_greek",
        "hellaswag_greek",
        "arc_easy_greek",
        "arc_challenge_greek",
        "ifeval_greek",
    }.issubset(TASK_REGISTRY.keys())


def test_num_samples_subset_is_stable_and_not_the_head():
    task = _RecordingTask(num_samples=50)
    with patch("pretrain.evaluation.base.trackio"):
        task.run(nn.Linear(1, 1), step=0)
        task.run(nn.Linear(1, 1), step=1)
    assert len(task.seen[0]) == 50
    assert task.seen[0] == task.seen[1]
    assert task.seen[0] != list(range(50))


def test_num_samples_larger_than_dataset_keeps_everything():
    task = _RecordingTask(num_samples=5000)
    with patch("pretrain.evaluation.base.trackio"):
        task.run(nn.Linear(1, 1), step=0)
    assert task.seen[0] == list(range(1000))


def test_runner_builds_one_task_instance_per_registered_task():
    config = EvaluationConfig(
        enabled=True,
        tasks={"mmlu": EvaluationTaskConfig(enabled=True, num_samples=5)},
    )
    runner = EvaluationRunner(config, tokenizer=MagicMock(), accelerator=MagicMock())
    assert len(runner.tasks) == len(TASK_REGISTRY)
