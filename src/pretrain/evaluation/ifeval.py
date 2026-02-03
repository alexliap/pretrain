"""IFEval instruction following benchmark."""

from typing import Any

from transformers import PreTrainedModel

from pretrain.evaluation.base import EvaluationTask


class IFEvalTask(EvaluationTask):
    """IFEval instruction following benchmark."""

    @property
    def name(self) -> str:
        return "ifeval"

    def load_data(self) -> list[dict[str, Any]]:
        """Load IFEval dataset."""
        # TODO: Implement IFEval data loading
        raise NotImplementedError("IFEval task not yet implemented")

    def evaluate_batch(
        self, model: PreTrainedModel, examples: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate IFEval instruction following accuracy."""
        # TODO: Implement IFEval evaluation
        raise NotImplementedError("IFEval task not yet implemented")
