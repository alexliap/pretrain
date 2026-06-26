import heapq
import os
import shutil
from pathlib import Path

from accelerate import Accelerator
from transformers import PreTrainedModel

from pretrain.config import TrainingConfig


class CheckpointManager:
    """Manages top-k checkpoints based on validation loss."""

    def __init__(self, save_dir: str, top_k: int):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            top_k: Number of best checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        # Max heap (negated losses): (-val_loss, step, path)
        # Worst checkpoint (largest real loss) is at index 0
        self.checkpoints: list[tuple[float, int, str]] = []

    def should_save(self, val_loss: float) -> bool:
        """Check if checkpoint should be saved based on validation loss.

        Args:
            val_loss: Current validation loss

        Returns:
            True if checkpoint should be saved
        """
        if len(self.checkpoints) < self.top_k:
            return True
        # If current loss is better (smaller) than worst kept checkpoint
        # self.checkpoints[0][0] is -worst_loss, so worst_loss = -self.checkpoints[0][0]
        worst_loss = -self.checkpoints[0][0]
        return val_loss < worst_loss

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        val_loss: float,
        step: int,
        config: TrainingConfig,
        accelerator: Accelerator,
    ) -> None:
        """Save checkpoint and manage top-k.

        Args:
            model: Model to save
            val_loss: Validation loss for this checkpoint
            step: Current training step
            config: Training configuration
            accelerator: Accelerator instance
        """
        checkpoint_name = f"checkpoint-step{step}-loss{val_loss:.4f}"
        checkpoint_path = self.save_dir / checkpoint_name

        # Save model using accelerator (handles distributed saving)
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(
                checkpoint_path,
                safe_serialization=True,
                max_shard_size=config.max_shard_size,
            )
            accelerator.print(f"Saved checkpoint: {checkpoint_path}")

        # Update checkpoint tracking (negate loss for max-heap behavior)
        heapq.heappush(self.checkpoints, (-val_loss, step, str(checkpoint_path)))

        # Remove worst checkpoint if exceeding top_k
        if len(self.checkpoints) > self.top_k:
            neg_worst_loss, _, worst_path = heapq.heappop(self.checkpoints)
            worst_loss = -neg_worst_loss
            if accelerator.is_main_process and os.path.exists(worst_path):
                shutil.rmtree(worst_path)
                accelerator.print(
                    f"Removed checkpoint: {worst_path} (loss: {worst_loss:.4f})"
                )

    def save_last_checkpoint(
        self,
        model: PreTrainedModel,
        step: int,
        config: TrainingConfig,
        accelerator: Accelerator,
    ) -> None:
        """Save the final ("last") model regardless of top-k ranking.

        Always written to a fixed ``last`` directory (overwriting any previous
        one) so it can be reliably picked up for resuming/further training. This
        checkpoint is not tracked in the top-k heap and is never pruned.

        Args:
            model: Model to save
            step: Current training step
            config: Training configuration
            accelerator: Accelerator instance
        """
        checkpoint_path = self.save_dir / "last"

        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            unwrapped_model.save_pretrained(
                checkpoint_path,
                safe_serialization=True,
                max_shard_size=config.max_shard_size,
            )
            accelerator.print(f"Saved last checkpoint: {checkpoint_path} (step {step})")

    def get_best_checkpoint(self) -> str | None:
        """Get path to best checkpoint.

        Returns:
            Path to checkpoint with lowest validation loss, or None if no checkpoints
        """
        if not self.checkpoints:
            return None
        # Find checkpoint with most negative value (smallest actual loss)
        best = min(self.checkpoints, key=lambda x: x[0])
        return best[2]
