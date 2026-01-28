"""Entry point for training."""

from accelerate.utils import set_seed

from pretrain.config import TrainingConfig
from pretrain.trainer import train

set_seed(0)


def main():
    """Run training with configured parameters."""
    config = TrainingConfig(
        tokenizer_path="el_en_tokenizer/tokenizers/tokenizer_64_0.1",
        base_model="Qwen/Qwen3-0.6B",
        hidden_size=128,
        intermediate_size=1024,
        learning_rate=1e-4,
        batch_size=16,
        num_epochs=1,
        num_workers=0,
        max_grad_norm=1.0,
        warmup_steps=200,
        val_check_interval=500,
        val_size=5000,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        project_name="test-project",
        auto_log_gpu=True,
    )

    train(config)


if __name__ == "__main__":
    main()
