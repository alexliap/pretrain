"""Training functions for pretraining."""

import gc
import os

import torch
import trackio
import yaml
from accelerate import Accelerator
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from pretrain.checkpoint import CheckpointManager
from pretrain.config import TrainingConfig
from pretrain.evaluation import EvaluationRunner


def setup_accelerator(config: TrainingConfig) -> Accelerator:
    """Initialize Accelerate accelerator with config.

    Args:
        config: Training configuration

    Returns:
        Configured Accelerator instance
    """
    return Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )


def load_model_and_tokenizer(
    config: TrainingConfig,
) -> tuple[PreTrainedModel, AutoConfig, AutoTokenizer]:
    """Load model, tokenizer, and config.

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, model_config, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    vocab_size = len(tokenizer)

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
            attn_implementation="sdpa",
        )
    else:
        print(f"Loading model from checkpoint: {config.saved_checkpoint_path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            config.saved_checkpoint_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        model_config = model.config

    # model = torch.compile(model)  # Disabled due to stride mismatch issue

    return model, model_config, tokenizer


def setup_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    from pretrain.model import PretrainDataLoader

    train_dataloader = PretrainDataLoader(
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        max_seq_length=config.max_seq_length,
        use_packed_data=config.use_packed_data,
    ).train_dataloader()

    val_dataloader = PretrainDataLoader(
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        max_seq_length=config.max_seq_length,
        use_packed_data=config.use_packed_data,
    ).val_dataloader(size=config.val_size)

    return train_dataloader, val_dataloader


def setup_optimizer_and_scheduler(
    model: PreTrainedModel, config: TrainingConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create optimizer and learning rate scheduler.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        eps=config.eps,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    scheduler = LinearLR(
        optimizer=optimizer,
        start_factor=config.warmup_start_factor,
        total_iters=config.warmup_steps,
    )

    return optimizer, scheduler


def print_model_info(
    model: PreTrainedModel, config: AutoConfig, accelerator: Accelerator
) -> None:
    """Print model summary information.

    Args:
        model: Model to summarize
        config: Model configuration
        accelerator: Accelerator instance for distributed printing
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print("\n" + "=" * 50)
    accelerator.print("MODEL SUMMARY")
    accelerator.print("=" * 50)
    accelerator.print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    accelerator.print(
        f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
    )
    accelerator.print(f"Vocab size: {config.vocab_size}")
    accelerator.print(f"Hidden size: {config.hidden_size}")
    accelerator.print(f"Intermediate size: {config.intermediate_size}")
    accelerator.print(f"Number of layers: {config.num_hidden_layers}")
    accelerator.print(f"Number of attention heads: {config.num_attention_heads}")
    accelerator.print("=" * 50 + "\n")


def compute_loss(model: PreTrainedModel, batch: dict) -> torch.Tensor:
    """Compute loss for a single batch.

    Args:
        model: Model to compute loss with
        batch: Batch of data containing input_ids

    Returns:
        Loss tensor
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")

    # Let the model do the causal shift internally. Mask padding positions in
    # the labels so loss is not computed on padding (validation batches).
    labels = input_ids.clone()
    if attention_mask is not None:
        labels[attention_mask == 0] = -100

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss if hasattr(outputs, "loss") else outputs


def training_step(
    model: PreTrainedModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    config: TrainingConfig,
) -> tuple[float, float, float]:
    """Execute single training step.

    Args:
        model: Model to train
        batch: Batch of data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        accelerator: Accelerator for distributed training
        config: Training configuration

    Returns:
        Tuple of (loss, current_lr, grad_norm)
    """
    with accelerator.accumulate(model):
        # Forward pass
        loss = compute_loss(model, batch)

        # Backward pass
        accelerator.backward(loss)

        # Gradient clipping and norm calculation
        grad_norm = 0.0
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        return loss.item(), current_lr, grad_norm.item()


def validate(
    model: PreTrainedModel,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
) -> float:
    """Run validation and return average loss.

    Args:
        model: Model to validate
        val_dataloader: Validation data loader
        accelerator: Accelerator for distributed training

    Returns:
        Average validation loss
    """
    model.eval()
    total_val_loss = 0.0

    val_progress_bar = tqdm(
        val_dataloader,
        desc="Validating",
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    with torch.no_grad():
        for val_batch in val_progress_bar:
            loss = compute_loss(model, val_batch).item()
            total_val_loss += loss
            val_progress_bar.set_postfix({"val_loss": f"{loss:.4f}"})

    avg_val_loss = total_val_loss / len(val_dataloader)
    model.train()

    return avg_val_loss


def train_epoch(
    epoch: int,
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    config: TrainingConfig,
    checkpoint_manager: CheckpointManager,
    evaluation_runner: EvaluationRunner | None = None,
    tokens_so_far: int = 0,
) -> int:
    """Train for one epoch with periodic validation and evaluation.

    Args:
        epoch: Current epoch number
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        accelerator: Accelerator for distributed training
        config: Training configuration
        checkpoint_manager: Manager for checkpoint saving
        evaluation_runner: Optional evaluation runner for benchmarks
        tokens_so_far: Cumulative tokens seen in previous epochs (for the
            total_tokens budget that spans across epochs)

    Returns:
        Tuple of (cumulative tokens seen after this epoch, number of steps run
        in this epoch).
    """
    model.train()
    cumulative_tokens = tokens_so_far
    val_loss = 0.0

    # Stopping-criterion hierarchy: total_tokens -> total_steps -> num_epochs.
    # The progress bar is driven by whichever criterion is active.
    if config.total_tokens is not None:
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=config.total_tokens,
            initial=tokens_so_far,
            unit="tok",
            unit_scale=True,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        )
    else:
        effective_steps = (
            config.total_steps
            if config.total_steps is not None
            else len(train_dataloader)
        )
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=effective_steps,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        )

    step = -1
    for step, batch in progress_bar:
        # Training step
        loss, current_lr, grad_norm = training_step(
            model, batch, optimizer, scheduler, accelerator, config
        )
        # Count the actual tokens in this batch (handles uneven batch sizes).
        batch_tokens = int(batch["input_ids"].numel())
        cumulative_tokens += batch_tokens

        # Advance the progress bar by tokens (token mode) or by step (otherwise).
        if config.total_tokens is not None:
            progress_bar.update(batch_tokens)

        # Log metrics
        if step % config.log_every_n == 0:
            trackio.log(
                {
                    "train_loss": round(loss, 4),
                    "learning_rate": current_lr,
                    "grad_norm": round(grad_norm, 4),
                    "tokens_passed": cumulative_tokens,
                    "iteration": step,
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
        if (step + 1) % config.val_check_interval == 0:
            gc.collect()
            val_loss = validate(model, val_dataloader, accelerator)
            trackio.log({"val_loss": round(val_loss, 4)})

            # Save checkpoint if it's in top-k
            if checkpoint_manager.should_save(val_loss):
                checkpoint_manager.save_checkpoint(
                    model, val_loss, step + 1, config, accelerator
                )

            # Run evaluations at same interval as validation
            if evaluation_runner is not None:
                evaluation_runner.run_all(model, step)

        # Save checkpoint every N steps (regardless of validation)
        if (step + 1) % config.save_every_n_steps == 0:
            # If we just validated, skip saving again
            if (step + 1) % config.val_check_interval != 0:
                # Use last validation loss or inf if no validation yet
                save_loss = val_loss if val_loss > 0.0 else float("inf")
                checkpoint_manager.save_checkpoint(
                    model, save_loss, step + 1, config, accelerator
                )

        # Stopping criteria (in priority order).
        if config.total_tokens is not None:
            if cumulative_tokens >= config.total_tokens:
                break
        elif config.total_steps is not None and (step + 1) >= config.total_steps:
            break

    progress_bar.close()
    return cumulative_tokens, step + 1


def train(config: TrainingConfig) -> None:
    """Main training function.

    Args:
        config: Training configuration
    """
    # Setup
    accelerator = setup_accelerator(config)
    model, model_config, tokenizer = load_model_and_tokenizer(config)
    print_model_info(model, model_config, accelerator)

    train_dataloader, val_dataloader = setup_dataloaders(config)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)

    # Prepare with accelerator
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )

    # Initialize checkpoint manager with model-specific and datetime-based subdirectory
    experiment_name = config.experiment_name or config.model.name
    save_dir = os.path.join(config.save_dir, experiment_name, config.run_name)
    checkpoint_manager = CheckpointManager(save_dir, config.save_top_k)

    # Save training config alongside checkpoints
    if accelerator.is_main_process:
        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config.get_dict(), f, default_flow_style=False)

    # Initialize evaluation runner
    evaluation_runner = None
    if config.evaluation.enabled:
        evaluation_runner = EvaluationRunner(
            config.evaluation,
            tokenizer,
            accelerator,
        )

    # Training loop
    tokens_seen = 0
    global_step = 0
    for epoch in range(config.num_epochs):
        tokens_seen, steps_in_epoch = train_epoch(
            epoch,
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            accelerator,
            config,
            checkpoint_manager,
            evaluation_runner,
            tokens_so_far=tokens_seen,
        )
        global_step += steps_in_epoch

        # Stop early once the token budget is exhausted (spans epochs).
        if config.total_tokens is not None and tokens_seen >= config.total_tokens:
            break

    # Always save the final model (regardless of top-k) for resuming/further
    # training. Lives in a fixed `last/` dir alongside the top-k checkpoints.
    if config.save_last:
        checkpoint_manager.save_last_checkpoint(
            model, global_step, config, accelerator
        )

    # Print best checkpoint
    best_checkpoint = checkpoint_manager.get_best_checkpoint()
    if best_checkpoint:
        accelerator.print(f"Best checkpoint: {best_checkpoint}")

    accelerator.print("Training complete!")
