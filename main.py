"""Entry point for training."""

import trackio
from accelerate.utils import set_seed

from pretrain.config import EvaluationConfig, EvaluationTaskConfig, TrainingConfig
from pretrain.trainer import train

set_seed(0)


def main():
    """Run training with configured parameters."""
    config = TrainingConfig(
        tokenizer_path="el_en_tokenizer/tokenizers/tokenizer_64_0.1",
        base_model="Qwen/Qwen3-0.6B",
        # saved_checkpoint_path="./checkpoints/checkpoint-step300-loss9.6758",
        hidden_size=128,
        intermediate_size=1024,
        learning_rate=5e-5,
        batch_size=4,
        num_epochs=1,
        total_steps=1_000,
        num_workers=0,
        max_grad_norm=1.0,
        warmup_steps=200,
        val_check_interval=250,
        val_size=15_000,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        max_seq_length=2048,  # Context window size for packing
        use_packed_data=True,  # Use packed dataset (run pack_data.py first)
        project_name="test-project",
        auto_log_gpu=True,
        save_dir="checkpoints",
        save_top_k=3,
        save_every_n_steps=100,
        max_shard_size="5GB",
        # Evaluation configuration (disabled by default)
        # Enable evaluations and implement tasks in src/pretrain/evaluation/ to use
        evaluation=EvaluationConfig(
            enabled=False,  # Set to True to enable evaluations
            humaneval=EvaluationTaskConfig(
                enabled=False,
                num_samples=50,  # Use subset for faster evaluation
                max_new_tokens=512,
                temperature=0.0,  # Greedy decoding
            ),
            mmlu=EvaluationTaskConfig(
                enabled=False,
                num_samples=100,
                temperature=0.0,
            ),
            ifeval=EvaluationTaskConfig(
                enabled=False,
                num_samples=50,
                max_new_tokens=512,
                temperature=0.0,
            ),
            log_predictions=False,  # Set to True to save results to files
            save_results_dir="eval_results",
        ),
    )

    # Initialize logging
    trackio.init(
        project=config.project_name,
        auto_log_gpu=config.auto_log_gpu,
        name=config.run_name,
        config=config.get_dict()
    )

    train(config)

    trackio.finish()


if __name__ == "__main__":
    main()
