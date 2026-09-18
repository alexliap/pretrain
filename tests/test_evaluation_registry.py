"""EvaluationRunner must build one task per registered TASK_REGISTRY entry,
with no network access needed at construction time."""

from unittest.mock import MagicMock

from pretrain.config import EvaluationConfig, EvaluationTaskConfig
from pretrain.evaluation import TASK_REGISTRY, EvaluationRunner


def test_registry_contains_expected_tasks():
    assert {"mmlu", "hellaswag", "piqa"}.issubset(TASK_REGISTRY.keys())


def test_runner_builds_one_task_instance_per_registered_task():
    config = EvaluationConfig(
        enabled=True,
        tasks={"mmlu": EvaluationTaskConfig(enabled=True, num_samples=5)},
    )
    runner = EvaluationRunner(config, tokenizer=MagicMock(), accelerator=MagicMock())
    assert len(runner.tasks) == len(TASK_REGISTRY)
