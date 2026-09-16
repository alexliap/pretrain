"""Direct Preference Optimization (DPO) via TRL's DPOTrainer."""

from pretrain.training.dpo.config import DPORunConfig
from pretrain.training.dpo.task import DPOTask

__all__ = ["DPORunConfig", "DPOTask"]
