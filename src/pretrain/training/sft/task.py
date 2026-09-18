"""Supervised fine-tuning workflow built on ``trl.SFTTrainer``.

This is a parallel path to :class:`pretrain.training.pretraining.task.PretrainTask`.
Rather than the hand-rolled Accelerate loop, it delegates the training loop,
data collation, prompt/completion loss masking, and checkpointing to TRL. The
model/tokenizer
load idiom mirrors ``PretrainTask._init_model_and_tokenizer`` (bf16 +
flash-attn kernel) and the LoRA wrapping reuses the same ``peft.LoraConfig``
shape as the pretraining path.
"""

import torch
import trackio
from accelerate import PartialState
from trl import SFTConfig, SFTTrainer

from pretrain.training.post_training_common import (
    TrackioCallback,
    build_peft_config,
    load_split,
    load_tokenizer,
)
from pretrain.training.sft.config import SFTRunConfig


class SFTTask:
    """End-to-end SFT workflow."""

    def __init__(self, config: SFTRunConfig):
        self.config = config

    def _build_sft_config(self) -> SFTConfig:
        cfg = self.config
        sft = cfg.sft

        # Let TRL load the model from the string path with our load-time kwargs
        # (bf16 + flash-attn kernel), matching the pretraining loader.
        model_init_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": cfg.attn_implementation,
        }

        # transformers 5.15 dropped `warmup_ratio` from TrainingArguments
        # entirely (not just deprecated it), so it can't be forwarded even as
        # None. Only pass it through when actually set, letting warmup_steps
        # win otherwise (see SFTArgsConfig.warmup_ratio's docstring).
        warmup_kwargs = {}
        if sft.warmup_ratio is not None:
            warmup_kwargs["warmup_ratio"] = sft.warmup_ratio

        return SFTConfig(
            output_dir=sft.output_dir,
            num_train_epochs=sft.num_train_epochs,
            max_steps=sft.max_steps,
            per_device_train_batch_size=sft.per_device_train_batch_size,
            per_device_eval_batch_size=sft.per_device_eval_batch_size,
            gradient_accumulation_steps=sft.gradient_accumulation_steps,
            learning_rate=sft.learning_rate,
            lr_scheduler_type=sft.lr_scheduler_type,
            warmup_steps=sft.warmup_steps,
            **warmup_kwargs,
            max_grad_norm=sft.max_grad_norm,
            weight_decay=sft.weight_decay,
            adam_beta1=sft.adam_beta1,
            adam_beta2=sft.adam_beta2,
            adam_epsilon=sft.adam_epsilon,
            max_length=sft.max_length,
            packing=sft.packing,
            dataset_num_proc=sft.dataset_num_proc,
            assistant_only_loss=sft.assistant_only_loss,
            completion_only_loss=sft.completion_only_loss,
            bf16=sft.bf16,
            gradient_checkpointing=sft.gradient_checkpointing,
            logging_steps=sft.logging_steps,
            save_steps=sft.save_steps,
            save_total_limit=sft.save_total_limit,
            eval_strategy=sft.eval_strategy,
            eval_steps=sft.eval_steps,
            # trackio is driven by our own TrackioCallback (see task docstring),
            # so the Trainer's built-in reporters stay off.
            report_to="none",
            run_name=cfg.run_name,
            model_init_kwargs=model_init_kwargs,
        )

    def train(self) -> None:
        """Run the SFT job end-to-end."""
        config = self.config

        # Only the main process talks to trackio - under DDP every process
        # runs this same code, and initializing on every rank would create
        # one trackio run per rank instead of one shared run.
        is_main_process = PartialState().is_main_process
        if is_main_process:
            trackio.init(
                project=config.logging.project_name,
                auto_log_gpu=config.logging.auto_log_gpu,
                name=config.run_name,
                config=config.get_dict(),
                space_id=None,
            )

        tokenizer = load_tokenizer(config.tokenizer_path, config.chat_template_path)
        train_dataset = load_split(
            config.dataset.dataset_id, config.dataset.dataset_split
        )
        if config.dataset.shuffle:
            train_dataset = train_dataset.shuffle(seed=config.dataset.shuffle_seed)
        eval_dataset = (
            load_split(config.dataset.dataset_id, config.dataset.eval_split)
            if config.dataset.eval_split
            else None
        )

        sft_config = self._build_sft_config()
        peft_config = build_peft_config(config.lora)

        trainer = SFTTrainer(
            model=config.model_name_or_path,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[TrackioCallback()],
        )

        trainer.train()

        trainer.save_model(config.sft.output_dir)
        trainer.save_state()

        if is_main_process:
            trackio.finish()
