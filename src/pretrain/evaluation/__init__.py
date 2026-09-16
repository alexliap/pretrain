"""Evaluation module for language model benchmarks."""

from pretrain.evaluation.base import EvaluationResult, EvaluationTask
from pretrain.evaluation.hellaswag import HellaSwagTask
from pretrain.evaluation.humaneval import HumanEvalTask
from pretrain.evaluation.ifeval import IFEvalTask
from pretrain.evaluation.mmlu import MMLUTask
from pretrain.evaluation.registry import TASK_REGISTRY, register_task
from pretrain.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationResult",
    "EvaluationTask",
    "HellaSwagTask",
    "HumanEvalTask",
    "IFEvalTask",
    "MMLUTask",
    "TASK_REGISTRY",
    "register_task",
    "EvaluationRunner",
]
