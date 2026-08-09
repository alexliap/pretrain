"""Configuration for supervised fine-tuning (SFT) runs.

A parallel config to :class:`pretrain.pretraining.config.TrainingConfig`, but
shaped around ``trl.SFTConfig``/``trl.SFTTrainer`` instead of the custom
pretraining loop. Built on pydantic ``BaseModel`` so a resolved Hydra dict
validates recursively in one call (``SFTRunConfig(**cfg_dict)``). The ``lora``
and ``logging`` sections are reused verbatim from the shared ``pretrain.config``
(they are stdlib dataclasses; pydantic coerces the incoming dicts into them) so
LoRA settings and trackio project naming stay consistent across pretraining,
CPT, and SFT.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, model_validator

from pretrain.config import LoggingConfig, LoraConfig


class DatasetConfig(BaseModel):
    """Configuration for the SFT dataset.

    ``dataset_id`` is either a Hugging Face Hub dataset id or a local path saved
    with ``datasets.save_to_disk`` (loaded via ``load_from_disk`` when it points
    at a directory on disk).

    ``dataset_format`` selects how TRL interprets rows:
      - ``"prompt_completion"``: rows of ``{"prompt": ..., "completion": ...}``.
        TRL masks the prompt automatically (loss on the completion only) with no
        chat-template edit, the recommended default.
      - ``"conversational"``: rows of ``{"messages": [...]}`` rendered with the
        tokenizer's chat template. True assistant-only loss additionally needs
        ``assistant_only_loss=True`` and a chat template with ``{% generation %}``
        markers.
      - ``"text"``: rows with a single ``dataset_text_field`` of raw text.
    """

    dataset_id: str = ""
    dataset_split: str = "train"
    eval_split: str | None = None
    dataset_format: str = "prompt_completion"
    dataset_text_field: str = "text"

    # One-time shuffle of the train split at startup (eval is left in order).
    # The Trainer also reshuffles each epoch, but this randomizes the initial
    # order once, useful for mixed-source data and for packing.
    shuffle: bool = True
    shuffle_seed: int = 0


class SFTArgsConfig(BaseModel):
    """Knobs forwarded onto ``trl.SFTConfig`` (a HF ``TrainingArguments``).

    Field names mirror ``SFTConfig`` so they can be splatted straight through.
    """

    output_dir: str = "checkpoints/sft"
    num_train_epochs: float = 1.0
    max_steps: int = -1  # -1 = derive from epochs; set >0 for smoke tests
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    # transformers 5.x: warmup_ratio is deprecated and, when not None, OVERWRITES
    # warmup_steps in TrainingArguments.__post_init__. Keep it None so warmup_steps
    # is honored. warmup_steps alone covers both modes: >=1 = absolute step count,
    # <1 = fraction of total steps.
    warmup_ratio: float | None = None
    warmup_steps: float = 0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0

    # SFT data handling
    max_length: int = 2048
    packing: bool = False
    assistant_only_loss: bool = False
    completion_only_loss: bool | None = None

    # Precision / memory
    bf16: bool = True
    gradient_checkpointing: bool = False

    # Logging / checkpointing (TRL's own, not the repo's CheckpointManager)
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int | None = 3
    eval_strategy: str = "no"  # set "steps" when an eval split is provided
    eval_steps: int = 500


class SFTRunConfig(BaseModel):
    """Top-level configuration for an SFT run.

    Mirrors the container-of-sections shape of ``TrainingConfig``. ``lora`` is
    optional: ``None`` means full-parameter SFT; a present section wraps the
    model with a fresh LoRA adapter (same convention as the pretraining path).
    """

    # Where to start SFT from: a merged-CPT dir or the raw base
    # (e.g. "models/lfm2_5_base/"). tokenizer_path defaults to model_name_or_path.
    model_name_or_path: str = ""
    tokenizer_path: str | None = None

    # Optional chat-template override (a .jinja file). For conversational SFT
    # with assistant-only loss, point this at a template containing
    # ``{% generation %}`` markers (the shipped LFM2 template has none), so TRL
    # can build the assistant token mask. None keeps the tokenizer's own template.
    chat_template_path: str | None = None

    dataset: DatasetConfig = DatasetConfig()
    sft: SFTArgsConfig = SFTArgsConfig()
    logging: LoggingConfig = LoggingConfig()

    # Optional LoRA adapter; None means full-parameter SFT.
    lora: LoraConfig | None = None

    # Model load-time kwargs (mirrors the pretraining loader).
    attn_implementation: str = "kernels-community/flash-attn2"

    # Name of the trackio run; defaults to a timestamp when unset.
    run_name: str | None = None

    @model_validator(mode="after")
    def _fill_defaults(self) -> "SFTRunConfig":
        if not self.model_name_or_path:
            raise ValueError("'model_name_or_path' must be provided for SFT")
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_name_or_path
        if not self.run_name:
            now = datetime.now(tz=UTC)
            self.run_name = f"sft-{now.date()}-{now.hour}-{now.minute}"
        return self

    def get_dict(self) -> dict:
        return self.model_dump()
