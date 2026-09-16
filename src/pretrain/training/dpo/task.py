"""Direct Preference Optimization workflow built on ``trl.DPOTrainer``.

This is a parallel path to :class:`pretrain.training.sft.task.SFTTask`. Rather than the
hand-rolled Accelerate loop, it delegates the training loop, data collation,
and reference-model handling to TRL. The model/tokenizer load idiom mirrors
``SFTTask`` (bf16 + flash-attn kernel) and the LoRA wrapping reuses the same
``peft.LoraConfig`` shape as the SFT/pretraining paths.
"""

import torch
import trackio
from accelerate import PartialState
from trl import DPOConfig, DPOTrainer

from pretrain.training.dpo.config import DPORunConfig
from pretrain.training.post_training_common import (
    TrackioCallback,
    build_peft_config,
    load_split,
    load_tokenizer,
)


class DPOTask:
    """End-to-end DPO workflow."""

    def __init__(self, config: DPORunConfig):
        self.config = config

    def _build_dpo_config(self) -> DPOConfig:
        cfg = self.config
        dpo = cfg.dpo

        # Let TRL load the model from the string path with our load-time kwargs
        # (bf16 + flash-attn kernel), matching the SFT/pretraining loader.
        model_init_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": cfg.attn_implementation,
        }

        # transformers 5.15 dropped `warmup_ratio` from TrainingArguments
        # entirely (not just deprecated it), so it can't be forwarded even as
        # None. Only pass it through when actually set, letting warmup_steps
        # win otherwise (see DPOArgsConfig.warmup_ratio's docstring).
        warmup_kwargs = {}
        if dpo.warmup_ratio is not None:
            warmup_kwargs["warmup_ratio"] = dpo.warmup_ratio

        return DPOConfig(
            output_dir=dpo.output_dir,
            num_train_epochs=dpo.num_train_epochs,
            max_steps=dpo.max_steps,
            per_device_train_batch_size=dpo.per_device_train_batch_size,
            per_device_eval_batch_size=dpo.per_device_eval_batch_size,
            gradient_accumulation_steps=dpo.gradient_accumulation_steps,
            learning_rate=dpo.learning_rate,
            lr_scheduler_type=dpo.lr_scheduler_type,
            warmup_steps=dpo.warmup_steps,
            **warmup_kwargs,
            max_grad_norm=dpo.max_grad_norm,
            weight_decay=dpo.weight_decay,
            adam_beta1=dpo.adam_beta1,
            adam_beta2=dpo.adam_beta2,
            adam_epsilon=dpo.adam_epsilon,
            beta=dpo.beta,
            loss_type=dpo.loss_type,
            label_smoothing=dpo.label_smoothing,
            max_length=dpo.max_length,
            truncation_mode=dpo.truncation_mode,
            disable_dropout=dpo.disable_dropout,
            precompute_ref_log_probs=dpo.precompute_ref_log_probs,
            dataset_num_proc=dpo.dataset_num_proc,
            bf16=dpo.bf16,
            gradient_checkpointing=dpo.gradient_checkpointing,
            logging_steps=dpo.logging_steps,
            save_steps=dpo.save_steps,
            save_total_limit=dpo.save_total_limit,
            eval_strategy=dpo.eval_strategy,
            eval_steps=dpo.eval_steps,
            # trackio is driven by our own TrackioCallback (see task docstring),
            # so the Trainer's built-in reporters stay off.
            report_to="none",
            run_name=cfg.run_name,
            model_init_kwargs=model_init_kwargs,
        )

    def train(self) -> None:
        """Run the DPO job end-to-end."""
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

        dpo_config = self._build_dpo_config()
        peft_config = build_peft_config(config.lora)

        trainer = DPOTrainer(
            model=config.model_name_or_path,
            ref_model=config.ref_model_name_or_path,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[TrackioCallback()],
        )

        trainer.train()

        trainer.save_model(config.dpo.output_dir)
        trainer.save_state()

        if is_main_process:
            trackio.finish()
