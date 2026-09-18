"""HellaSwag commonsense-completion benchmark."""

import re
from typing import Any

from datasets import load_dataset
from transformers import PreTrainedModel

from pretrain.evaluation.base import EvaluationTask
from pretrain.evaluation.registry import register_task
from pretrain.evaluation.scoring import accuracy_from_choices

_BRACKET_TAG_RE = re.compile(r"\[.*?\]")


def _clean_text(text: str) -> str:
    """Strip lm-eval-harness-style bracketed section tags (e.g. "[header]")
    from raw HellaSwag ctx/ending text and normalize whitespace."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = _BRACKET_TAG_RE.sub("", text)
    text = text.replace("  ", " ")
    return text.strip()


@register_task("hellaswag")
class HellaSwagTask(EvaluationTask):
    """HellaSwag commonsense-completion benchmark."""

    @property
    def name(self) -> str:
        return "hellaswag"

    def load_data(self) -> list[dict[str, Any]]:
        """Load HellaSwag dataset (validation split - the test split has no
        public labels)."""
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        examples = []
        for row in dataset:
            ctx = _clean_text(row["ctx_a"] + " " + row["ctx_b"].capitalize())
            endings = [_clean_text(ending) for ending in row["endings"]]
            examples.append(
                {
                    "context": ctx,
                    "choices": [f" {ending}" for ending in endings],
                    "gold": int(row["label"]),
                }
            )
        return examples

    def evaluate_batch(
        self, model: PreTrainedModel, examples: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate HellaSwag accuracy via loglikelihood over each ending."""
        return accuracy_from_choices(
            model, self.tokenizer, examples, self.config.batch_size
        )
