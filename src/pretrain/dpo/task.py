"""Direct Preference Optimization workflow built on ``trl.DPOTrainer``.

This is a parallel path to :class:`pretrain.sft.task.SFTTask`. Rather than the
hand-rolled Accelerate loop, it delegates the training loop, data collation,
and reference-model handling to TRL. The model/tokenizer load idiom mirrors
``SFTTask`` (bf16 + flash-attn kernel) and the LoRA wrapping reuses the same
``peft.LoraConfig`` shape as the SFT/pretraining paths.
"""

import os

import torch
import trackio
from accelerate import PartialState
from datasets import DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig as PeftLoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

from pretrain.dpo.config import DPORunConfig


class TrackioCallback(TrainerCallback):
    """Log TRL/Trainer metrics to the trackio run started in ``DPOTask.train``.

    We don't use transformers' built-in ``report_to="trackio"`` integration: in
    transformers 5.12.1 its callback calls ``trackio.init(bucket_id=...)``, an
    argument the installed trackio doesn't accept. This callback logs the same
    metrics against the run already initialized in the entrypoint instead.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if metrics:
            trackio.log(metrics, step=state.global_step)


class DPOTask:
    """End-to-end DPO workflow."""

    def __init__(self, config: DPORunConfig):
        self.config = config

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        # DPO needs a pad token for batching; fall back to eos when absent.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Optional chat-template override (e.g. one matching the SFT stage's).
        if self.config.chat_template_path is not None:
            with open(self.config.chat_template_path) as f:
                tokenizer.chat_template = f.read()
        return tokenizer

    def _load_split(self, split: str):
        """Load a dataset split from the Hub, a local ``save_to_disk`` dir, or a
        local raw-file directory (e.g. a ``hf download`` snapshot of a Hub
        dataset repo, with data files under ``data/`` and no
        ``dataset_dict.json``/``dataset_info.json`` marker)."""
        dataset_id = self.config.dataset.dataset_id
        is_saved_to_disk = os.path.isfile(
            os.path.join(dataset_id, "dataset_dict.json")
        ) or os.path.isfile(os.path.join(dataset_id, "dataset_info.json"))
        if is_saved_to_disk:
            dataset = load_from_disk(dataset_id)
            # load_from_disk yields a DatasetDict for saved splits, or a bare
            # Dataset when a single split was saved.
            return dataset[split] if isinstance(dataset, DatasetDict) else dataset

        return load_dataset(dataset_id, split=split)

    def _build_peft_config(self) -> PeftLoraConfig | None:
        """Build a peft LoRA config from the reused LoraConfig, or None for full FT."""
        lora = self.config.lora
        if lora is None:
            return None

        return PeftLoraConfig(
            r=lora.r,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            target_modules=lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

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

        tokenizer = self._load_tokenizer()
        train_dataset = self._load_split(config.dataset.dataset_split)
        if config.dataset.shuffle:
            train_dataset = train_dataset.shuffle(seed=config.dataset.shuffle_seed)

        eval_dataset = (
            self._load_split(config.dataset.eval_split)
            if config.dataset.eval_split
            else None
        )

        dpo_config = self._build_dpo_config()
        peft_config = self._build_peft_config()

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
