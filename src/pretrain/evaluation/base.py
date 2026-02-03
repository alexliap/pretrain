"""Base classes for evaluation tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import trackio
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizer

from pretrain.config import EvaluationTaskConfig


@dataclass
class EvaluationResult:
    """Result from an evaluation task."""

    task_name: str
    metrics: dict[str, float]  # e.g., {"pass@1": 0.45, "accuracy": 0.78}
    num_samples: int
    step: int


class EvaluationTask(ABC):
    """Base class for evaluation tasks."""

    def __init__(
        self,
        task_config: EvaluationTaskConfig,
        tokenizer: PreTrainedTokenizer,
        accelerator: Accelerator,
    ):
        self.config = task_config
        self.tokenizer = tokenizer
        self.accelerator = accelerator

    @property
    @abstractmethod
    def name(self) -> str:
        """Task name for logging."""
        pass

    @abstractmethod
    def load_data(self) -> list[dict[str, Any]]:
        """Load evaluation dataset.

        Returns:
            List of evaluation examples.
        """
        pass

    @abstractmethod
    def evaluate_batch(
        self, model: PreTrainedModel, examples: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate model on a batch of examples.

        Args:
            model: Model to evaluate
            examples: Batch of evaluation examples

        Returns:
            Dictionary of metrics
        """
        pass

    def run(self, model: PreTrainedModel, step: int) -> EvaluationResult | None:
        """Run full evaluation.

        Args:
            model: Model to evaluate
            step: Current training step

        Returns:
            Evaluation result with metrics, or None if task is disabled
        """
        if not self.config.enabled:
            return None

        try:
            self.accelerator.print(f"\n{'='*50}")
            self.accelerator.print(f"Running {self.name} evaluation...")
            self.accelerator.print(f"{'='*50}")

            model.eval()
            data = self.load_data()

            # Limit samples if configured
            if self.config.num_samples:
                data = data[: self.config.num_samples]

            with torch.no_grad():
                metrics = self.evaluate_batch(model, data)

            model.train()

            result = EvaluationResult(
                task_name=self.name,
                metrics=metrics,
                num_samples=len(data),
                step=step,
            )

            self._log_result(result)
            return result

        except Exception as e:
            self.accelerator.print(f"ERROR in {self.name} evaluation: {e}")
            trackio.log({f"eval/{self.name}/error": 1})
            model.train()  # Ensure model is back in training mode
            return None

    def _log_result(self, result: EvaluationResult):
        """Log evaluation result to trackio."""
        log_dict = {
            f"eval/{result.task_name}/{k}": v for k, v in result.metrics.items()
        }
        log_dict[f"eval/{result.task_name}/num_samples"] = result.num_samples

        trackio.log(log_dict)

        # Print to console
        self.accelerator.print(f"\n{result.task_name} Results:")
        for metric, value in result.metrics.items():
            self.accelerator.print(f"  {metric}: {value:.4f}")
