"""MMLU knowledge benchmark (cloze-style: loglikelihood over full answer text,
no A/B/C/D letters - matches this repo's existing eval_results/REPORT.md
convention for "MMLU cloze")."""

from typing import Any

from datasets import load_dataset
from transformers import PreTrainedModel

from pretrain.evaluation.base import EvaluationTask
from pretrain.evaluation.registry import register_task
from pretrain.evaluation.scoring import accuracy_from_choices


@register_task("mmlu")
class MMLUTask(EvaluationTask):
    """MMLU knowledge benchmark, cloze-style loglikelihood scoring."""

    DATASET_ID = "cais/mmlu"

    @property
    def name(self) -> str:
        return "mmlu"

    def load_data(self) -> list[dict[str, Any]]:
        """Load MMLU dataset (all 57 subjects, test split)."""
        dataset = load_dataset(self.DATASET_ID, "all", split="test")
        return [
            {
                "context": f"Question: {row['question']}\nAnswer:",
                "choices": [f" {choice}" for choice in row["choices"]],
                "gold": row["answer"],
            }
            for row in dataset
        ]

    def evaluate_batch(
        self, model: PreTrainedModel, examples: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate MMLU accuracy via loglikelihood over each answer choice."""
        return accuracy_from_choices(
            model, self.tokenizer, examples, self.config.batch_size
        )


@register_task("mmlu_greek")
class MMLUGreekTask(MMLUTask):
    """Greek translation of MMLU (ilsp/mmlu_greek). Same "all"-config schema
    and scoring as MMLUTask; unlike HellaSwag/PIQA, the test split has real
    labels, matching the English task's choice of split."""

    DATASET_ID = "ilsp/mmlu_greek"

    @property
    def name(self) -> str:
        return "mmlu_greek"
