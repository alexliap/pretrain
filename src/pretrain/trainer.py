"""Training functions for pretraining."""

from typing import Tuple

import torch
import trackio
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

from pretrain.config import TrainingConfig


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
    config: TrainingConfig, accelerator: Accelerator
) -> Tuple[PreTrainedModel, AutoTokenizer, AutoConfig]:
    """Load model, tokenizer, and config.

    Args:
        config: Training configuration
        accelerator: Accelerator instance for distributed training

    Returns:
        Tuple of (model, tokenizer, model_config)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    vocab_size = len(tokenizer)

    # Create model config
    model_config = AutoConfig.from_pretrained(config.base_model)
    model_config.hidden_size = config.hidden_size
    model_config.intermediate_size = config.intermediate_size
    model_config.vocab_size = vocab_size

    # Create model
    model = AutoModelForCausalLM.from_config(model_config, dtype=torch.bfloat16)
    model = torch.compile(model)

    accelerator.print(f"Loaded tokenizer from {config.tokenizer_path}")

    return model, tokenizer, model_config


def setup_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    from pretrain.model import MyData

    train_dataloader = MyData(
        num_workers=config.num_workers, batch_size=config.batch_size
    ).train_dataloader()

    val_dataloader = MyData(
        num_workers=config.num_workers, batch_size=config.batch_size
    ).val_dataloader(size=config.val_size)

    return train_dataloader, val_dataloader


def setup_optimizer_and_scheduler(
    model: PreTrainedModel, config: TrainingConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create optimizer and learning rate scheduler.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, eps=1e-10
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
    x = batch["input_ids"][:, :-1]
    y = batch["input_ids"][:, 1:]
    outputs = model(x, labels=y)
    return outputs.loss if hasattr(outputs, "loss") else outputs


def training_step(
    model: PreTrainedModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    config: TrainingConfig,
) -> Tuple[float, float]:
    """Execute single training step.

    Args:
        model: Model to train
        batch: Batch of data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        accelerator: Accelerator for distributed training
        config: Training configuration

    Returns:
        Tuple of (loss, current_lr)
    """
    with accelerator.accumulate(model):
        # Forward pass
        loss = compute_loss(model, batch)

        # Backward pass
        accelerator.backward(loss)

        # Gradient clipping
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        return loss.item(), current_lr


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
    total_steps: int,
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    config: TrainingConfig,
) -> None:
    """Train for one epoch with periodic validation.

    Args:
        epoch: Current epoch number
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        accelerator: Accelerator for distributed training
        config: Training configuration
    """
    # Initialize logging
    trackio.init(project=config.project_name, auto_log_gpu=config.auto_log_gpu)

    model.train()
    total_tokens_passed = 0
    val_loss = 0.0

    progress_bar = tqdm(
        enumerate(train_dataloader),
        total=total_steps,
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch + 1}/{config.num_epochs}",
    )

    for step, batch in progress_bar:
        # Training step
        loss, current_lr = training_step(
            model, batch, optimizer, scheduler, accelerator, config
        )
        total_tokens_passed += int(
            torch.prod(torch.tensor(batch["input_ids"].size())).item()
        )
        # Log metrics
        if step % config.log_every_n == 0:
            trackio.log(
                {
                    "train_loss": round(loss, 4),
                    "learning_rate": current_lr,
                    "tokens_passed": total_tokens_passed,
                    "iteration": step
                }
            )

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "val_loss": f"{val_loss:.4f}",
            }
        )

        # Periodic validation
        if (step + 1) % config.val_check_interval == 0:
            val_loss = validate(model, val_dataloader, accelerator)
            trackio.log({"val_loss": round(val_loss, 4)})


def train(config: TrainingConfig) -> None:
    """Main training function.

    Args:
        config: Training configuration
    """
    # Setup
    accelerator = setup_accelerator(config)
    model, tokenizer, model_config = load_model_and_tokenizer(config, accelerator)
    print_model_info(model, model_config, accelerator)

    train_dataloader, val_dataloader = setup_dataloaders(config)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)

    # Prepare with accelerator
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )

    # Training loop
    for epoch in range(config.num_epochs):
        train_epoch(
            epoch,
            config.total_steps,
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            accelerator,
            config,
        )

    accelerator.print("Training complete!")
