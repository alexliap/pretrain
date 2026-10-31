"""Evaluation module for language model benchmarks."""

from pretrain.evaluation.arc import (
    ArcChallengeGreekTask,
    ArcChallengeTask,
    ArcEasyGreekTask,
    ArcEasyTask,
)
from pretrain.evaluation.base import EvaluationResult, EvaluationTask
from pretrain.evaluation.hellaswag import HellaSwagGreekTask, HellaSwagTask
from pretrain.evaluation.humaneval import HumanEvalTask
from pretrain.evaluation.ifeval import IFEvalGreekTask, IFEvalTask
from pretrain.evaluation.mmlu import MMLUGreekTask, MMLUTask
from pretrain.evaluation.piqa import PIQATask
from pretrain.evaluation.registry import TASK_REGISTRY, register_task
from pretrain.evaluation.runner import EvaluationRunner

__all__ = [
    "ArcChallengeGreekTask",
    "ArcChallengeTask",
    "ArcEasyGreekTask",
    "ArcEasyTask",
    "EvaluationResult",
    "EvaluationTask",
    "HellaSwagGreekTask",
    "HellaSwagTask",
    "HumanEvalTask",
    "IFEvalGreekTask",
    "IFEvalTask",
    "MMLUGreekTask",
    "MMLUTask",
    "PIQATask",
    "TASK_REGISTRY",
    "register_task",
    "EvaluationRunner",
]
