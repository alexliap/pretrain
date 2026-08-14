"""Entry point for Direct Preference Optimization (DPO)."""

import hydra
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf

from pretrain.dpo import DPORunConfig, DPOTask

set_seed(0)


@hydra.main(version_base=None, config_path="configs", config_name="dpo")
def main(cfg: DictConfig) -> None:
    """Run DPO with Hydra configuration."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = DPORunConfig(**cfg_dict)

    # Trackio init/finish and all logging happen inside DPOTask, gated to the
    # main process only - under DDP every process runs this script, and
    # initializing here unconditionally would create one trackio run per rank.
    DPOTask(config).train()


if __name__ == "__main__":
    main()
