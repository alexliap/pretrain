"""PIQA physical-commonsense benchmark (cloze-style: loglikelihood over each
solution's full text, matching this repo's existing eval_results/REPORT.md
"PIQA (acc_norm)" convention and the harness's piqa.yaml doc_to_text/doc_to_choice)."""

from typing import Any

from datasets import load_dataset
from transformers import PreTrainedModel

from pretrain.evaluation.base import EvaluationTask
from pretrain.evaluation.registry import register_task
from pretrain.evaluation.scoring import accuracy_from_choices


@register_task("piqa")
class PIQATask(EvaluationTask):
    """PIQA physical-commonsense benchmark, cloze-style loglikelihood scoring."""

    @property
    def name(self) -> str:
        return "piqa"

    def load_data(self) -> list[dict[str, Any]]:
        """Load PIQA dataset (validation split - the test split has no public
        labels). Uses baber/piqa, a parquet re-upload of the original
        piqa/ybisk/piqa dataset: the original ships a HF loading script, and
        datasets==5.0.1 (this repo's pinned version) dropped default
        script-execution support, so baber/piqa avoids needing
        trust_remote_code=True."""
        dataset = load_dataset("baber/piqa", split="validation")
        return [
            {
                "context": f"Question: {row['goal']}\nAnswer:",
                "choices": [f" {row['sol1']}", f" {row['sol2']}"],
                "gold": row["label"],
            }
            for row in dataset
        ]

    def evaluate_batch(
        self, model: PreTrainedModel, examples: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate PIQA accuracy via loglikelihood over each solution choice."""
        return accuracy_from_choices(
            model, self.tokenizer, examples, self.config.batch_size
        )
