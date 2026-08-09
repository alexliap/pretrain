"""Entry point for training."""

import hydra
from accelerate.utils import set_seed
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from pretrain.pretraining import PretrainTask, TrainingConfig

load_dotenv()
set_seed(0)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run training with Hydra configuration."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = TrainingConfig.from_dict(cfg_dict)

    # Trackio init/finish and all logging happen inside PretrainTask, gated to
    # the main process only - under DDP every process runs this script, and
    # initializing here unconditionally would create one trackio run per rank.
    PretrainTask(config).train()


if __name__ == "__main__":
    main()
