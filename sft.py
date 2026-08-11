"""Entry point for supervised fine-tuning (SFT)."""

import hydra
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf

from pretrain.sft import SFTRunConfig, SFTTask

set_seed(0)


@hydra.main(version_base=None, config_path="configs", config_name="sft")
def main(cfg: DictConfig) -> None:
    """Run SFT with Hydra configuration."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = SFTRunConfig(**cfg_dict)

    # Trackio init/finish and all logging happen inside SFTTask, gated to the
    # main process only - under DDP every process runs this script, and
    # initializing here unconditionally would create one trackio run per rank.
    SFTTask(config).train()


if __name__ == "__main__":
    main()
