"""Configuration for training."""

from dataclasses import dataclass, asdict, field
from datetime import datetime


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
class TrainingConfig:
    """Configuration for training run.
    When saved_checkpoint_path is given model configuration parameters are overriden and not used at all.
    If saved_checkpoint_path is None, then model config has to be set up."""

    # Model config
    tokenizer_path: str
    base_model: str | None = None
    hidden_size: int = 128
    intermediate_size: int = 1024

    saved_checkpoint_path: str | None = None

    # Training config
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 1
    total_steps: int = 10000
    num_workers: int = 0
    max_grad_norm: float = 1.0

    # Optimizer config
    eps: float = 1e-10
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1

    # Scheduler config
    warmup_steps: int = 200

    # Validation config
    val_check_interval: int = 500
    val_size: int = 10000

    # Accelerate config
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"

    # Logging config
    project_name: str = "test-project"
    auto_log_gpu: bool = True
    log_every_n: int = 5

    # Checkpoint saving config
    save_dir: str = "checkpoints"
    save_top_k: int = 3
    save_every_n_steps: int = 1000
    max_shard_size: str = "5GB"

    # Evaluation config
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        if self.base_model is None and self.saved_checkpoint_path is None:
            raise ValueError(
                "At least one of 'base_model' or 'saved_checkpoint_path' must be provided"
            )

    @property
    def warmup_start_factor(self):
        return self.learning_rate / self.warmup_steps

    @property
    def run_name(self):
        run_name = ""
        run_name += str(datetime.now().date())
        run_name += (
            "-" + str(str(datetime.now().hour)) + "-" + str(str(datetime.now().minute))
        )
        return run_name
    
    def get_dict(self):
        return asdict(self)
