"""ARC (AI2 Reasoning Challenge) science-QA benchmark, Easy and Challenge
subsets (cloze-style: loglikelihood over full answer text, matching this
repo's existing eval_results/REPORT.md "ARC (acc_norm)" convention)."""

from typing import Any

from datasets import load_dataset
from transformers import PreTrainedModel

from pretrain.evaluation.base import EvaluationTask
from pretrain.evaluation.registry import register_task
from pretrain.evaluation.scoring import accuracy_from_choices


class ArcTask(EvaluationTask):
    """Shared loading/scoring for the ARC-Easy and ARC-Challenge configs."""

    DATASET_ID = "allenai/ai2_arc"
    CONFIG_NAME = "ARC-Easy"

    def load_data(self) -> list[dict[str, Any]]:
        """Load an ARC config's test split (real labels; unlike hellaswag/
        piqa, ARC's test split does ship answerKeys)."""
        dataset = load_dataset(self.DATASET_ID, self.CONFIG_NAME, split="test")
        examples = []
        for row in dataset:
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            examples.append(
                {
                    "context": f"Question: {row['question']}\nAnswer:",
                    "choices": [f" {text}" for text in texts],
                    "gold": labels.index(row["answerKey"]),
                }
            )
        return examples

    def evaluate_batch(
        self, model: PreTrainedModel, examples: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate ARC accuracy via loglikelihood over each answer choice."""
        return accuracy_from_choices(
            model, self.tokenizer, examples, self.config.batch_size
        )


@register_task("arc_easy")
class ArcEasyTask(ArcTask):
    """ARC-Easy: the subset of ARC questions that don't require the
    ARC-Challenge questions' harder reasoning/retrieval."""

    CONFIG_NAME = "ARC-Easy"

    @property
    def name(self) -> str:
        return "arc_easy"


@register_task("arc_challenge")
class ArcChallengeTask(ArcTask):
    """ARC-Challenge: the harder subset (wrong by both a retrieval-based and
    a co-occurrence-based baseline)."""

    CONFIG_NAME = "ARC-Challenge"

    @property
    def name(self) -> str:
        return "arc_challenge"


@register_task("arc_easy_greek")
class ArcEasyGreekTask(ArcEasyTask):
    """Greek translation of ARC-Easy (ilsp/arc_greek). Same schema and
    scoring; test split has real labels, same as the English version."""

    DATASET_ID = "ilsp/arc_greek"

    @property
    def name(self) -> str:
        return "arc_easy_greek"


@register_task("arc_challenge_greek")
class ArcChallengeGreekTask(ArcChallengeTask):
    """Greek translation of ARC-Challenge (ilsp/arc_greek)."""

    DATASET_ID = "ilsp/arc_greek"

    @property
    def name(self) -> str:
        return "arc_challenge_greek"
