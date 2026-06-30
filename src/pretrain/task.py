import gc
import json
import os

import torch
import trackio
import yaml
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from pretrain.checkpoint import CheckpointManager, TrainingState
from pretrain.config import TrainingConfig
from pretrain.dataloader import PretrainDataLoader
from pretrain.evaluation import EvaluationRunner


class PretrainTask:
    """End-to-end pretraining workflow.

    All setup steps populate instance state (``self.model``, ``self.optimizer``,
    ...) rather than returning values. :meth:`train` is the orchestrator that
    wires the steps together and runs the training loop.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def _init_accelerator(self) -> None:
        """Initialize the Accelerate accelerator from config."""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.accelerate.gradient_accumulation_steps,
            mixed_precision=self.config.accelerate.mixed_precision,
        )

    def _init_model_and_tokenizer(self) -> None:
        """Load model, tokenizer, and model config onto the task instance.

        Sets:
            self.model, self.model_config, self.tokenizer
        """
        config = self.config

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
        vocab_size = len(tokenizer)

        resuming = config.saved_checkpoint_path is not None and os.path.isdir(
            os.path.join(config.saved_checkpoint_path, "state")
        )

        if config.saved_checkpoint_path is None:
            # Create model config
            model_config = AutoConfig.from_pretrained(config.model.base_model)
            model_config.hidden_size = config.model.hidden_size
            model_config.intermediate_size = config.model.intermediate_size
            model_config.head_dim = config.model.head_dim
            model_config.num_hidden_layers = config.model.num_hidden_layers
            model_config.num_heads = config.model.num_heads
            model_config.vocab_size = vocab_size

            model = AutoModelForCausalLM.from_config(
                model_config,
                dtype=torch.bfloat16,
                attn_implementation="kernels-community/flash-attn2",
            )
        elif resuming:
            # Resuming: only the architecture is needed here; the actual weights are
            # restored later by accelerator.load_state from the saved training state.
            # Building from config (not from_pretrained) means we don't depend on HF
            # weights being present in the checkpoint dir — just its config.json.
            #
            # A LoRA checkpoint holds only the adapter (adapter_config.json, no
            # config.json), so the base architecture config is read from the
            # adapter's base_model_name_or_path instead of the checkpoint dir.
            adapter_config_path = os.path.join(
                config.saved_checkpoint_path, "adapter_config.json"
            )
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path) as f:
                    config_source = json.load(f)["base_model_name_or_path"]
            else:
                config_source = config.saved_checkpoint_path

            print(
                f"Resuming: building model architecture from {config_source} ..."
            )
            model_config = AutoConfig.from_pretrained(config_source)
            model = AutoModelForCausalLM.from_config(
                model_config,
                dtype=torch.bfloat16,
                attn_implementation="kernels-community/flash-attn2",
            )
        else:
            print(f"Loading model from checkpoint: {config.saved_checkpoint_path} ...")
            model = AutoModelForCausalLM.from_pretrained(
                config.saved_checkpoint_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            )
            model_config = model.config

        # Optionally wrap the base model with a LoRA adapter. Done uniformly for
        # all build paths and before compile/prepare. On the resume path the
        # adapter starts fresh here but is overwritten by accelerator.load_state.
        if config.lora is not None:
            lora_cfg = PeftLoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.lora_alpha,
                lora_dropout=config.lora.lora_dropout,
                target_modules=config.lora.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        if self.config.compile == "torch":
            torch.set_float32_matmul_precision("high")
            model = torch.compile(model)  # Disabled due to stride mismatch issue

        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer

    def _init_dataloader(self) -> None:
        """Build the dataloader factory shared by train/validation loaders."""
        config = self.config
        self.dataloader = PretrainDataLoader(
            num_workers=config.data.num_workers,
            train_batch_size=config.data.train_batch_size,
            val_batch_size=config.data.val_batch_size,
            max_seq_length=config.data.max_seq_length,
            use_packed_data=config.data.use_packed_data,
        )

    def _init_train_dataloader(self) -> None:
        """Create the training dataloader (sets self.train_dataloader)."""
        self.train_dataloader = self.dataloader.train_dataloader()

    def _init_validation_dataloader(self) -> None:
        """Create the validation dataloader (sets self.val_dataloader)."""
        self.val_dataloader = self.dataloader.val_dataloader(
            size=self.config.validation.val_size
        )

    def _init_optimizer_and_scheduler(self) -> None:
        """Create the optimizer and LR scheduler (sets self.optimizer/self.scheduler)."""
        config = self.config

        self.optimizer = torch.optim.AdamW(
            # Only trainable params — under LoRA this is just the adapter; for
            # full training every param requires grad, so this is a no-op.
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.optimizer.learning_rate,
            eps=config.optimizer.eps,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )

        self.scheduler = LinearLR(
            optimizer=self.optimizer,
            start_factor=config.warmup_start_factor,
            total_iters=config.scheduler.warmup_steps,
        )

    def _print_model_info(self) -> None:
        """Print model summary information."""
        model_config = self.model_config
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self.accelerator.print("\n" + "=" * 50)
        self.accelerator.print("MODEL SUMMARY")
        self.accelerator.print("=" * 50)
        self.accelerator.print(
            f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)"
        )
        self.accelerator.print(
            f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
        )
        self.accelerator.print(f"Vocab size: {model_config.vocab_size}")
        self.accelerator.print(f"Hidden size: {model_config.hidden_size}")
        self.accelerator.print(f"Intermediate size: {model_config.intermediate_size}")
        self.accelerator.print(f"Number of layers: {model_config.num_hidden_layers}")
        self.accelerator.print(
            f"Number of attention heads: {model_config.num_attention_heads}"
        )
        self.accelerator.print("=" * 50 + "\n")

    def _compute_loss(self, batch: dict, model=None) -> torch.Tensor:
        """Compute loss for a single batch.

        Args:
            batch: Batch of data containing input_ids
            model: Model to run the forward pass with. Defaults to self.model
                (compiled). Validation passes the eager model to avoid recompiles
                on its variable-length, masked batches.

        Returns:
            Loss tensor
        """
        model = model if model is not None else self.model

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Let the model do the causal shift internally. Mask padding positions in
        # the labels so loss is not computed on padding (validation batches).
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss if hasattr(outputs, "loss") else outputs

    def _training_step(self, batch: dict) -> tuple[float, float, float]:
        """Execute a single training step.

        Args:
            batch: Batch of data

        Returns:
            Tuple of (loss, current_lr, grad_norm)
        """
        with self.accelerator.accumulate(self.model):
            # Forward pass
            loss = self._compute_loss(batch)

            # Backward pass
            self.accelerator.backward(loss)

            # Gradient clipping and norm calculation
            grad_norm = 0.0
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.max_grad_norm
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]

            return loss.item(), current_lr, grad_norm.item()

    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_val_loss = 0.0

        # Run validation on the eager (un-compiled) model. Validation batches are
        # padded and carry an attention mask, which would otherwise force the
        # compiled model to recompile/graph-break every batch. unwrap_model strips
        # the DDP wrapper; _orig_mod strips the torch.compile wrapper if present.
        eval_model = self.accelerator.unwrap_model(self.model)
        eval_model = getattr(eval_model, "_orig_mod", eval_model)

        # The eager loss path materializes full float32 [B, T, vocab] logits that
        # the compiled training path fuses away. Release the cached training pool
        # so that allocation has room (otherwise validation can OOM even though
        # training fit).
        gc.collect()
        torch.cuda.empty_cache()

        val_progress_bar = tqdm(
            self.val_dataloader,
            desc="Validating",
            disable=not self.accelerator.is_local_main_process,
            leave=False,
        )

        with torch.no_grad():
            for val_batch in val_progress_bar:
                loss = self._compute_loss(val_batch, model=eval_model).item()
                total_val_loss += loss
                val_progress_bar.set_postfix({"val_loss": f"{loss:.4f}"})

        avg_val_loss = total_val_loss / len(self.val_dataloader)

        # Average across processes so every rank agrees on the validation loss.
        # Each rank only sees its shard of the (prepared) val dataloader, so the
        # per-rank averages differ. Reducing here gives the true val-set loss and,
        # crucially, keeps it identical on every rank — checkpoint directory names
        # embed the loss, so divergent values make non-main ranks write orphaned
        # state-only checkpoint dirs that never get pruned.
        avg_val_loss = self.accelerator.reduce(
            torch.tensor(avg_val_loss, device=self.accelerator.device),
            reduction="mean",
        ).item()
        self.model.train()

        # Free the validation allocations before resuming the compiled training
        # loop so its memory pool isn't fragmented by eager-validation buffers.
        gc.collect()
        torch.cuda.empty_cache()

        return avg_val_loss

    def _train_epoch(
        self,
        epoch: int,
        train_dataloader: DataLoader,
        start_step: int = 0,
    ) -> None:
        """Train for one epoch with periodic validation and evaluation.

        Args:
            epoch: Current epoch number
            train_dataloader: Training data loader (may already be fast-forwarded
                past consumed batches when resuming, see ``start_step``)
            start_step: Number of steps already completed in this epoch (non-zero
                only when resuming mid-epoch). ``train_dataloader`` is expected to
                have skipped these batches already, so it re-enumerates from 0 and
                the true step is ``start_step + local_step``.
        """
        config = self.config
        training_state = self.training_state

        self.model.train()
        cumulative_tokens = training_state.tokens_seen
        val_loss = 0.0

        # Stopping-criterion hierarchy: total_tokens -> total_steps -> num_epochs.
        # The progress bar is driven by whichever criterion is active.
        if config.total_tokens is not None:
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=config.total_tokens,
                initial=cumulative_tokens,
                unit="tok",
                unit_scale=True,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            )
        else:
            effective_steps = (
                config.total_steps
                if config.total_steps is not None
                else len(train_dataloader) + start_step
            )
            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=effective_steps,
                initial=start_step,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            )

        stopped = False
        for local_step, batch in progress_bar:
            # True step within the epoch (the loader re-enumerates from 0 even when
            # it was fast-forwarded past `start_step` batches on resume).
            step = start_step + local_step

            # Training step
            loss, current_lr, grad_norm = self._training_step(batch)
            # Count the actual tokens in this batch (handles uneven batch sizes).
            batch_tokens = int(batch["input_ids"].numel())
            cumulative_tokens += batch_tokens

            # Advance the progress bar by tokens (token mode) or by step (otherwise).
            if config.total_tokens is not None:
                progress_bar.update(batch_tokens)

            # Update progress counters (persisted via accelerator.save_state).
            training_state.epoch = epoch
            training_state.global_step += 1
            training_state.step_in_epoch = step + 1
            training_state.tokens_seen = cumulative_tokens

            # Log metrics
            if step % config.logging.log_every_n == 0:
                trackio.log(
                    {
                        "train_loss": round(loss, 4),
                        "learning_rate": current_lr,
                        "grad_norm": round(grad_norm, 4),
                        "tokens_passed": cumulative_tokens,
                        "iteration": training_state.global_step,
                    }
                )

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "grad_norm": f"{grad_norm:.2e}",
                    "val_loss": f"{val_loss:.4f}",
                }
            )

            # Periodic validation
            if (step + 1) % config.validation.val_check_interval == 0:
                val_loss = self._validate()
                trackio.log({"val_loss": round(val_loss, 4)})

                # Save checkpoint if it's in top-k (state is embedded inside it)
                if self.checkpoint_manager.should_save(val_loss):
                    self.checkpoint_manager.save_checkpoint(
                        self.model, val_loss, step + 1, config, self.accelerator
                    )

                # Run evaluations at same interval as validation
                if self.evaluation_runner is not None:
                    self.evaluation_runner.run_all(self.model, step)

            # Save checkpoint every N steps (regardless of validation). Each
            # checkpoint embeds its own resumable state (see save_checkpoint), so a
            # crashed run can resume from the latest surviving checkpoint.
            if (step + 1) % config.checkpoint.save_every_n_steps == 0:
                # If we just validated, skip the top-k save (already done above)
                if (step + 1) % config.validation.val_check_interval != 0:
                    # Use last validation loss or inf if no validation yet
                    save_loss = val_loss if val_loss > 0.0 else float("inf")
                    self.checkpoint_manager.save_checkpoint(
                        self.model, save_loss, step + 1, config, self.accelerator
                    )

            # Stopping criteria (in priority order).
            if config.total_tokens is not None:
                if cumulative_tokens >= config.total_tokens:
                    stopped = True
                    break
            elif config.total_steps is not None and (step + 1) >= config.total_steps:
                stopped = True
                break

        progress_bar.close()

        # If the epoch ran to its natural end (not cut short by a stop criterion),
        # advance to the next epoch so a resume doesn't re-run it.
        if not stopped:
            training_state.epoch = epoch + 1
            training_state.step_in_epoch = 0

    def _init_checkpointing(self) -> None:
        """Set up the checkpoint manager, persist config, and prepare evaluation.

        Sets:
            self.checkpoint_manager, self.evaluation_runner, self.training_state
        """
        config = self.config

        # run_name uses wall-clock time and is evaluated per-process; broadcast the
        # main process's value so every rank writes into the SAME run directory
        # (otherwise ranks that cross a minute boundary split into separate dirs).
        run_name = [config.checkpoint.run_name]
        broadcast_object_list(run_name, from_process=0)
        run_name = run_name[0]

        # Checkpoint manager with model-specific and datetime-based subdirectory
        experiment_name = config.checkpoint.experiment_name or config.model.name
        save_dir = os.path.join(config.checkpoint.save_dir, experiment_name, run_name)
        self.checkpoint_manager = CheckpointManager(
            save_dir, config.checkpoint.save_top_k, self.tokenizer
        )

        # Save training config alongside checkpoints
        if self.accelerator.is_main_process:
            config_path = os.path.join(save_dir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config.get_dict(), f, default_flow_style=False)

        # Initialize evaluation runner
        self.evaluation_runner = None
        if config.evaluation.enabled:
            self.evaluation_runner = EvaluationRunner(
                config.evaluation,
                self.tokenizer,
                self.accelerator,
            )

        # Progress counters, persisted with model/optimizer/scheduler via save_state.
        self.training_state = TrainingState()
        self.accelerator.register_for_checkpointing(self.training_state)

    def _maybe_resume(self) -> bool:
        """Restore training state when resuming from a checkpoint.

        Resume only when loading from a checkpoint that carries a saved state dir.
        Otherwise this is a fresh run / further-training from weights only.

        Returns:
            True if the run resumed from a saved state, False otherwise.
        """
        config = self.config
        if config.saved_checkpoint_path is not None:
            state_dir = os.path.join(config.saved_checkpoint_path, "state")
            if os.path.isdir(state_dir):
                self.accelerator.load_state(state_dir)
                self.accelerator.print(
                    f"Resumed from {state_dir}: epoch {self.training_state.epoch}, "
                    f"global_step {self.training_state.global_step}, "
                    f"tokens {self.training_state.tokens_seen}"
                )
                return True
        return False

    def _log_token_distribution(self) -> None:
        """Log the per-source token distribution recorded at tokenization time."""
        dist_path = "tokenized_data/token_distribution.json"
        if not os.path.exists(dist_path):
            self.accelerator.print(
                f"No token distribution found at {dist_path}; skipping."
            )
            return

        with open(dist_path) as f:
            distribution = json.load(f)

        total_tokens = distribution.get("total_tokens", 0)
        tokens_per_source = distribution.get("tokens_per_source", {})
        percentages = distribution.get("percentages", {})

        # Console summary (printed once, on the main process).
        self.accelerator.print("\n" + "=" * 50)
        self.accelerator.print("TOKEN DISTRIBUTION")
        self.accelerator.print("=" * 50)
        self.accelerator.print(f"Total tokens: {total_tokens:,}")
        for source, pct in sorted(
            percentages.items(), key=lambda kv: kv[1], reverse=True
        ):
            n = tokens_per_source.get(source, 0)
            self.accelerator.print(f"  {source:<20} {n:>15,} ({pct:.1%})")
        self.accelerator.print("=" * 50 + "\n")

        # Save the distribution file to the trackio project's files directory.
        if self.accelerator.is_main_process:
            trackio.save(dist_path)

    def _save_config(self) -> None:
        """Save the resolved run config to the trackio project's files directory."""
        if not self.accelerator.is_main_process:
            return

        config_path = "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(self.config.get_dict(), f, sort_keys=False)
        try:
            trackio.save(config_path)
        finally:
            os.remove(config_path)

    def train(self) -> None:
        """Orchestrate the full pretraining run."""
        config = self.config

        # Setup
        self._init_accelerator()
        self._init_model_and_tokenizer()
        self._print_model_info()

        self._init_dataloader()
        self._init_train_dataloader()
        self._init_validation_dataloader()
        self._log_token_distribution()
        self._save_config()
        self._init_optimizer_and_scheduler()

        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataloader,
            self.val_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataloader,
            self.val_dataloader,
        )

        # Checkpointing, evaluation, and resumable state
        self._init_checkpointing()
        resuming = self._maybe_resume()

        # Training loop
        for epoch in range(self.training_state.epoch, config.num_epochs):
            # On the first (resumed) epoch, fast-forward past consumed batches so no
            # sample is seen twice; later epochs start from the beginning.
            skip = (
                self.training_state.step_in_epoch
                if (resuming and epoch == self.training_state.epoch)
                else 0
            )
            loader = (
                self.accelerator.skip_first_batches(self.train_dataloader, skip)
                if skip
                else self.train_dataloader
            )

            self._train_epoch(epoch, loader, start_step=skip)

            # Stop early once the token budget is exhausted (spans epochs).
            if (
                config.total_tokens is not None
                and self.training_state.tokens_seen >= config.total_tokens
            ):
                break

        # Always save the final model (regardless of top-k) for resuming/further
        # training. Lives in a fixed `last/` dir alongside the top-k checkpoints and
        # embeds its own resumable state (see save_last_checkpoint).
        if config.checkpoint.save_last:
            self.checkpoint_manager.save_last_checkpoint(
                self.model, self.training_state.global_step, config, self.accelerator
            )

        # Print best checkpoint
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            self.accelerator.print(f"Best checkpoint: {best_checkpoint}")

        self.accelerator.print("Training complete!")
