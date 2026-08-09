"""Configuration for (continued-)pretraining runs.

The counterpart to :class:`pretrain.sft.config.SFTRunConfig`: everything the
hand-rolled Accelerate loop in ``task.py`` needs. The sections that are not
specific to this path, ``logging``, ``lora``, ``evaluation``, live in
``pretrain.config`` and are reused here, so SFT and the evaluation tasks can
import them without depending on this module.
"""

import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from pretrain.config import (
    EvaluationConfig,
    EvaluationTaskConfig,
    LoggingConfig,
    LoraConfig,
)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    name: str = ""
    base_model: str | None = None
    hidden_size: int = 128
    intermediate_size: int = 1024
    head_dim: int = 128
    num_hidden_layers: int = 28
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    max_position_embeddings: int | None = None


@dataclass
class DataConfig:
    """Configuration for the data/tokenization pipeline."""

    tokenizer_path: str = ""
    max_seq_length: int = 2048  # Context window size for packing
    use_packed_data: bool = True  # Whether to use packed dataset
    num_workers: int = 0
    train_batch_size: int = 16
    val_batch_size: int = 4


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer (AdamW) and gradient clipping."""

    learning_rate: float = 1e-4
    eps: float = 1e-10
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0


@dataclass
class SchedulerConfig:
    """Configuration for the learning rate scheduler."""

    warmup_steps: int = 200


@dataclass
class AccelerateConfig:
    """Configuration passed to the Accelerate accelerator."""

    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"


@dataclass
class ValidationConfig:
    """Configuration for periodic validation."""

    val_check_interval: int = 500
    val_size: int | None = 10000


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving."""

    save_dir: str = "checkpoints"
    # Name of the checkpoint subdirectory. Defaults to model.name when None,
    # which is only meaningful when building a model from the model config
    # group. Set this explicitly (e.g. when resuming from saved_checkpoint_path)
    # so the run dir isn't tied to an unused model config name.
    experiment_name: str | None = None
    run_name: str | None = None
    save_top_k: int = 3
    save_every_n_steps: int = 1000
    save_last: bool = True
    max_shard_size: str = "5GB"


@dataclass
class TrainingConfig:
    """Configuration for a training run.

    A thin container of per-concern sub-configs (``optimizer``, ``logging``, ...)
    plus the run-level fields that don't belong to a single section.

    When ``saved_checkpoint_path`` is given the model architecture parameters in
    ``model`` are overridden by the checkpoint's config and not used. If
    ``saved_checkpoint_path`` is None, then ``model`` has to be set up.
    """

    # Per-section sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    accelerate: AccelerateConfig = field(default_factory=AccelerateConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Optional LoRA adapter; None means full-parameter training.
    lora: LoraConfig | None = None

    # Run-level fields (kept top-level on purpose)
    saved_checkpoint_path: str | None = None
    compile: str = "torch"
    # Passed straight to AutoModelForCausalLM.from_config/from_pretrained. Defaults
    # to PyTorch's built-in "sdpa" since custom Hub kernels (e.g.
    # "kernels-community/flash-attn2/3") may not ship a compiled kernel image for
    # newer GPU architectures (confirmed missing for Blackwell/sm_120).
    attn_implementation: str = "sdpa"
    num_epochs: int = 1
    total_steps: int | None = None
    total_tokens: int | None = None

    def __post_init__(self):
        if self.model.base_model is None and self.saved_checkpoint_path is None:
            raise ValueError(
                "At least one of 'model.base_model' or 'saved_checkpoint_path' must be provided"
            )

        # Default the run name to a timestamp when not set explicitly in config.
        # Set checkpoint.run_name in the config to override with a custom name.
        if not self.checkpoint.run_name:
            now = datetime.now(tz=UTC)
            self.checkpoint.run_name = f"{now.date()}-{now.hour}-{now.minute}"

    @property
    def is_resuming(self) -> bool:
        """True when this run continues a checkpoint's saved training state.

        Mirrors the condition in ``PretrainTask._maybe_resume``: a resume only
        happens when ``saved_checkpoint_path`` carries a ``state`` dir. Loading
        weights without that dir is a fresh run, not a resume — so logging should
        continue the existing experiment-tracker run only in the former case.
        """
        if self.saved_checkpoint_path is None:
            return False
        return os.path.isdir(os.path.join(self.saved_checkpoint_path, "state"))

    @property
    def warmup_start_factor(self):
        return self.optimizer.learning_rate / self.scheduler.warmup_steps

    def get_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TrainingConfig":
        """Create TrainingConfig from a nested dictionary (e.g., from Hydra)."""
        # Pop each nested section and reconstruct its dataclass. Remaining keys
        # (the run-level fields) are unpacked into TrainingConfig directly.
        section_types = {
            "model": ModelConfig,
            "data": DataConfig,
            "optimizer": OptimizerConfig,
            "scheduler": SchedulerConfig,
            "accelerate": AccelerateConfig,
            "validation": ValidationConfig,
            "logging": LoggingConfig,
            "checkpoint": CheckpointConfig,
        }
        sections = {name: cls(**d.pop(name, {})) for name, cls in section_types.items()}

        # `lora` is optional: absent or null means full training. A present
        # section (even empty) enables LoRA, with defaults for unspecified fields.
        lora_dict = d.pop("lora", None)
        sections["lora"] = LoraConfig(**lora_dict) if lora_dict is not None else None

        eval_dict = d.pop("evaluation", {})
        if eval_dict:
            for task_name in ["humaneval", "ifeval", "mmlu"]:
                if task_name in eval_dict and isinstance(eval_dict[task_name], dict):
                    eval_dict[task_name] = EvaluationTaskConfig(**eval_dict[task_name])
            sections["evaluation"] = EvaluationConfig(**eval_dict)
        else:
            sections["evaluation"] = EvaluationConfig()

        return TrainingConfig(**sections, **d)
