"""Evaluation runner that coordinates all tasks."""

import json
from pathlib import Path

from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizer

from pretrain.config import EvaluationConfig
from pretrain.evaluation.base import EvaluationResult, EvaluationTask
from pretrain.evaluation.humaneval import HumanEvalTask
from pretrain.evaluation.ifeval import IFEvalTask
from pretrain.evaluation.mmlu import MMLUTask


class EvaluationRunner:
    """Coordinates running all enabled evaluation tasks."""

    def __init__(
        self,
        config: EvaluationConfig,
        tokenizer: PreTrainedTokenizer,
        accelerator: Accelerator,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.accelerator = accelerator

        # Initialize all tasks
        self.tasks: list[EvaluationTask] = [
            HumanEvalTask(config.humaneval, tokenizer, accelerator),
            IFEvalTask(config.ifeval, tokenizer, accelerator),
            MMLUTask(config.mmlu, tokenizer, accelerator),
        ]

    def run_all(
        self, model: PreTrainedModel, step: int
    ) -> list[EvaluationResult]:
        """Run all enabled evaluation tasks.

        Args:
            model: Model to evaluate
            step: Current training step

        Returns:
            List of evaluation results
        """
        if not self.config.enabled:
            return []

        results = []
        for task in self.tasks:
            if task.config.enabled:
                result = task.run(model, step)
                if result:
                    results.append(result)

        # Save results if configured
        if self.config.log_predictions and results:
            self._save_results(results, step)

        return results

    def _save_results(self, results: list[EvaluationResult], step: int):
        """Save evaluation results to file."""
        save_dir = Path(self.config.save_results_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            filename = save_dir / f"{result.task_name}_step{step}.json"
            with open(filename, "w") as f:
                json.dump(
                    {
                        "step": result.step,
                        "metrics": result.metrics,
                        "num_samples": result.num_samples,
                    },
                    f,
                    indent=2,
                )
