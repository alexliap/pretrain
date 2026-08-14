"""Configuration for Direct Preference Optimization (DPO) runs.

A parallel config to :class:`pretrain.sft.config.SFTRunConfig`, but shaped
around ``trl.DPOConfig``/``trl.DPOTrainer`` instead of ``trl.SFTTrainer``.
Built on pydantic ``BaseModel`` so a resolved Hydra dict validates
recursively in one call (``DPORunConfig(**cfg_dict)``). The ``lora`` and
``logging`` sections are reused verbatim from the shared ``pretrain.config``
so LoRA settings and trackio project naming stay consistent across
pretraining, SFT, and DPO.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, model_validator

from pretrain.config import LoggingConfig, LoraConfig


class DatasetConfig(BaseModel):
    """Configuration for the DPO preference dataset.

    ``dataset_id`` is either a Hugging Face Hub dataset id or a local path
    saved with ``datasets.save_to_disk`` (loaded via ``load_from_disk`` when
    it points at a directory on disk).

    ``dataset_format`` selects how TRL interprets rows:
      - ``"conversational"``: rows of ``{"prompt": [...], "chosen": [...],
        "rejected": [...]}`` as message lists, rendered with the tokenizer's
        chat template (the same template the SFT checkpoint DPO starts from
        was trained with).
      - ``"standard"``: rows of ``{"prompt": str, "chosen": str, "rejected": str}``.
    """

    dataset_id: str = ""
    dataset_split: str = "train"
    eval_split: str | None = None
    dataset_format: str = "conversational"

    # One-time shuffle of the train split at startup (eval is left in order).
    # The Trainer also reshuffles each epoch, but this randomizes the initial
    # order once, useful for mixed-source data.
    shuffle: bool = True
    shuffle_seed: int = 0


class DPOArgsConfig(BaseModel):
    """Knobs forwarded onto ``trl.DPOConfig`` (a HF ``TrainingArguments``).

    Field names mirror ``DPOConfig`` so they can be splatted straight through.
    """

    output_dir: str = "checkpoints/dpo"
    num_train_epochs: float = 1.0
    max_steps: int = -1  # -1 = derive from epochs; set >0 for smoke tests
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-7  # DPO learning rates run well below SFT's
    lr_scheduler_type: str = "cosine"
    # transformers 5.x: warmup_ratio is deprecated and, when not None, OVERWRITES
    # warmup_steps in TrainingArguments.__post_init__. Keep it None so warmup_steps
    # is honored. warmup_steps alone covers both modes: >=1 = absolute step count,
    # <1 = fraction of total steps.
    warmup_ratio: float | None = None
    warmup_steps: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-10

    # DPO loss
    beta: float = 0.1
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    max_length: int = 2048
    truncation_mode: str = "keep_start"
    # Disables dropout in both policy and reference models so repeated forward
    # passes over the same input are deterministic (trl's own default).
    disable_dropout: bool = True
    precompute_ref_log_probs: bool = False
    dataset_num_proc: int | None = 16

    # Precision / memory
    bf16: bool = True
    gradient_checkpointing: bool = False

    # Logging / checkpointing (TRL's own, not the repo's CheckpointManager)
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int | None = 3
    eval_strategy: str = "no"  # set "steps" when an eval split is provided
    eval_steps: int = 500


class DPORunConfig(BaseModel):
    """Top-level configuration for a DPO run.

    Mirrors the container-of-sections shape of ``SFTRunConfig``. ``lora`` is
    optional: ``None`` means full-parameter DPO; a present section wraps the
    model with a fresh LoRA adapter (same convention as SFT/pretraining).
    """

    # Where to start DPO from: typically the finished SFT checkpoint.
    # tokenizer_path defaults to model_name_or_path.
    model_name_or_path: str = ""
    tokenizer_path: str | None = None

    # Explicit reference model; None lets DPOTrainer build one from
    # model_name_or_path automatically (or use the PEFT disable-adapter trick
    # instead of a second copy, when `lora` is set).
    ref_model_name_or_path: str | None = None

    # Optional chat-template override (a .jinja file). None keeps the
    # tokenizer's own template (e.g. the one SFT already installed).
    chat_template_path: str | None = None

    dataset: DatasetConfig = DatasetConfig()
    dpo: DPOArgsConfig = DPOArgsConfig()
    logging: LoggingConfig = LoggingConfig()

    # Optional LoRA adapter; None means full-parameter DPO.
    lora: LoraConfig | None = None

    # Model load-time kwargs (mirrors the SFT/pretraining loader).
    attn_implementation: str = "kernels-community/flash-attn2"

    # Name of the trackio run; defaults to a timestamp when unset.
    run_name: str | None = None

    @model_validator(mode="after")
    def _fill_defaults(self) -> "DPORunConfig":
        if not self.model_name_or_path:
            raise ValueError("'model_name_or_path' must be provided for DPO")
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_name_or_path
        if not self.run_name:
            now = datetime.now(tz=UTC)
            self.run_name = f"dpo-{now.date()}-{now.hour}-{now.minute}"
        return self

    def get_dict(self) -> dict:
        return self.model_dump()
