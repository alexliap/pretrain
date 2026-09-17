"""Config sections shared across training paths.

Only the sections more than one module needs live here: ``logging`` and ``lora``
are used by both ``pretrain.pretraining`` and ``pretrain.sft``, and the
``evaluation`` sections are consumed by ``pretrain.evaluation``. Keeping them
out of ``pretrain.pretraining.config`` is what lets SFT and the evaluation tasks
reuse them without importing the pretraining module.
"""

from dataclasses import dataclass, field


@dataclass
class EvaluationTaskConfig:
    """Configuration for a single evaluation task."""

    enabled: bool = False
    num_samples: int | None = None  # None = use all samples
    batch_size: int = 1
    temperature: float = 0.0  # Greedy decoding
    max_new_tokens: int = 512


@dataclass
class EvaluationConfig:
    """Configuration for evaluation tasks."""

    enabled: bool = False  # Master switch

    # Task configs
    humaneval: EvaluationTaskConfig = field(default_factory=EvaluationTaskConfig)
    ifeval: EvaluationTaskConfig = field(default_factory=EvaluationTaskConfig)
    mmlu: EvaluationTaskConfig = field(default_factory=EvaluationTaskConfig)

    # Logging
    log_predictions: bool = False
    save_results_dir: str = "eval_results"


@dataclass
class LoggingConfig:
    """Configuration for experiment logging (trackio)."""

    project_name: str = "test-project"
    auto_log_gpu: bool = True
    log_every_n: int = 5


@dataclass
class LoraConfig:
    """LoRA adapter settings (maps onto peft's LoraConfig).

    Present this section to train with a LoRA adapter; leave it unset (the
    default ``None`` on TrainingConfig) for full-parameter training.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Defaults to the attention query/value projections. Set to None to let peft
    # auto-infer the target modules for the architecture.
    target_modules: list[str] | None = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    init_lora_weights: str = "lora"
