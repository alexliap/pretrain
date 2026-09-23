"""IFEval instruction-following benchmark.

Generate-then-verify, unlike the loglikelihood-scored tasks (mmlu, hellaswag,
piqa): the model free-generates a chat response to an instruction-laden
prompt, and a deterministic checker (pretrain.evaluation.ifeval_lib, vendored
from lm-evaluation-harness / Google's original IFEval) verifies whether the
response actually obeys each instruction. Scoring matches
eval_results/typakos-140m-it-dpo's offline lm-eval benchmark run, which used
the same reference implementation.
"""

from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedModel

from pretrain.evaluation.base import EvaluationTask
from pretrain.evaluation.ifeval_lib.utils import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from pretrain.evaluation.registry import register_task


@register_task("ifeval")
class IFEvalTask(EvaluationTask):
    """IFEval instruction following benchmark."""

    DATASET_ID = "google/IFEval"

    @property
    def name(self) -> str:
        return "ifeval"

    def load_data(self) -> list[dict[str, Any]]:
        """Load IFEval dataset (single "train" split, 541 prompts - the only
        split the dataset ships)."""
        dataset = load_dataset(self.DATASET_ID, split="train")
        return [
            {
                "key": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            }
            for row in dataset
        ]

    def _generate(self, model: PreTrainedModel, prompt: str) -> str:
        device = next(model.parameters()).device
        tokenizer = self.tokenizer
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        input_ids = inputs["input_ids"]

        do_sample = self.config.temperature > 0
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_token_ids = [tokenizer.eos_token_id]
        if eot_token_id is not None and eot_token_id != tokenizer.unk_token_id:
            eos_token_ids.append(eot_token_id)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=do_sample,
            temperature=self.config.temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
        )
        completion_ids = output_ids[0, input_ids.shape[1] :]
        return tokenizer.decode(completion_ids, skip_special_tokens=True)

    def evaluate_batch(
        self, model: PreTrainedModel, examples: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Generate a response per prompt and score it against its
        instruction_id_list/kwargs, strict and loose. Generation is one
        prompt at a time (variable-length, chat-templated) rather than
        batched, matching scripts/typakos_140m/test_chat_generation.py."""
        prompt_strict: list[bool] = []
        prompt_loose: list[bool] = []
        inst_strict: list[bool] = []
        inst_loose: list[bool] = []

        with torch.no_grad():
            for example in examples:
                response = self._generate(model, example["prompt"])
                inp = InputExample(
                    key=example["key"],
                    instruction_id_list=example["instruction_id_list"],
                    prompt=example["prompt"],
                    kwargs=example["kwargs"],
                )
                out_strict = test_instruction_following_strict(inp, response)
                out_loose = test_instruction_following_loose(inp, response)

                prompt_strict.append(out_strict.follow_all_instructions)
                prompt_loose.append(out_loose.follow_all_instructions)
                inst_strict.extend(out_strict.follow_instruction_list)
                inst_loose.extend(out_loose.follow_instruction_list)

        return {
            "prompt_level_strict_acc": sum(prompt_strict) / len(prompt_strict),
            "inst_level_strict_acc": sum(inst_strict) / len(inst_strict),
            "prompt_level_loose_acc": sum(prompt_loose) / len(prompt_loose),
            "inst_level_loose_acc": sum(inst_loose) / len(inst_loose),
        }


@register_task("ifeval_greek")
class IFEvalGreekTask(IFEvalTask):
    """Greek translation of IFEval (ilsp/ifeval_greek): same keys/
    instruction_id_list/kwargs as google/IFEval, row-for-row, only "prompt"
    is translated ("prompt_en" keeps the English original, unused here).
    Scoring is entirely inherited - the checker is language-agnostic string/
    format logic except language:response_language, which uses langdetect
    and works the same way for Greek ("el") responses."""

    DATASET_ID = "ilsp/ifeval_greek"

    @property
    def name(self) -> str:
        return "ifeval_greek"
