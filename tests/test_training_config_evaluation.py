"""TrainingConfig.from_dict must round-trip the evaluation.tasks dict shape
into EvaluationTaskConfig instances - the only automated check of this, since
Hydra's --cfg job/--help flags print and exit before from_dict runs."""

from pretrain.training.pretraining.config import TrainingConfig


def test_from_dict_builds_evaluation_tasks_dict():
    d = {
        "model": {"base_model": "x"},
        "evaluation": {
            "enabled": True,
            "tasks": {
                "mmlu": {"enabled": True, "num_samples": 5},
                "hellaswag": {"enabled": True, "num_samples": 5, "batch_size": 2},
            },
        },
    }
    config = TrainingConfig.from_dict(d)
    assert config.evaluation.tasks["mmlu"].num_samples == 5
    assert config.evaluation.tasks["hellaswag"].batch_size == 2
