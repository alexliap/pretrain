"""Helpers shared between the TRL-based training paths (SFT, DPO).

Both pretrain.training.sft.task.SFTTask and pretrain.training.dpo.task.DPOTask
wrap a trl Trainer around a common model/tokenizer/dataset loading idiom. This
module holds the pieces that were identical (or near-identical) between the
two: the trackio logging callback, tokenizer/dataset loading, and the peft
LoRA config builder. Each task's _build_*_config/train bodies stay in their
own module since DPOConfig/SFTConfig and DPOTrainer/SFTTrainer differ
meaningfully - only these stateless pieces are shared.
"""

import os

import trackio
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig as PeftLoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizer, TrainerCallback

from pretrain.config import LoraConfig


class TrackioCallback(TrainerCallback):
    """Log TRL/Trainer metrics to the trackio run started in the task's train().

    We don't use transformers' built-in ``report_to="trackio"`` integration: in
    transformers 5.12.1 its callback calls ``trackio.init(bucket_id=...)``, an
    argument the installed trackio doesn't accept. This callback logs the same
    metrics against the run already initialized in the entrypoint instead.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if metrics:
            trackio.log(metrics, step=state.global_step)


def load_tokenizer(
    tokenizer_path: str, chat_template_path: str | None = None
) -> PreTrainedTokenizer:
    """Load a tokenizer, defaulting pad_token to eos_token when absent, with an
    optional chat-template override (e.g. one matching a prior stage's)."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if chat_template_path is not None:
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
    return tokenizer


def load_split(dataset_id: str, split: str) -> Dataset:
    """Load a dataset split from the Hub, a local ``save_to_disk`` dir, or a
    local raw-file directory (e.g. a ``hf download`` snapshot of a Hub
    dataset repo, with data files under ``data/`` and no
    ``dataset_dict.json``/``dataset_info.json`` marker)."""
    is_saved_to_disk = os.path.isfile(
        os.path.join(dataset_id, "dataset_dict.json")
    ) or os.path.isfile(os.path.join(dataset_id, "dataset_info.json"))
    if is_saved_to_disk:
        dataset = load_from_disk(dataset_id)
        # load_from_disk yields a DatasetDict for saved splits, or a bare
        # Dataset when a single split was saved.
        return dataset[split] if isinstance(dataset, DatasetDict) else dataset
    # Hub id or a local directory of raw data files: load_dataset handles
    # both the same way, auto-detecting splits from filename patterns
    # (train-*, validation-*, ...).
    return load_dataset(dataset_id, split=split)


def build_peft_config(lora: LoraConfig | None) -> PeftLoraConfig | None:
    """Build a peft LoRA config from the shared LoraConfig, or None for full FT."""
    if lora is None:
        return None
    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
