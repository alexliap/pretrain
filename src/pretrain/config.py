"""Configuration for training."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training run."""

    # Model config
    tokenizer_path: str
    base_model: str = "Qwen/Qwen3-0.6B"
    hidden_size: int = 128
    intermediate_size: int = 1024

    # Training config
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 1
    total_steps: int = 10000
    num_workers: int = 0
    max_grad_norm: float = 1.0

    # Scheduler config
    warmup_steps: int = 200
    warmup_start_factor: float = 0.005  # lr / 200

    # Validation config
    val_check_interval: int = 500
    val_size: int = 5000

    # Accelerate config
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"

    # Logging config
    project_name: str = "test-project"
    auto_log_gpu: bool = True
    log_every_n: int = 5
