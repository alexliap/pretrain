"""Entry point for training."""

import hydra
import trackio
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf

from pretrain.config import TrainingConfig
from pretrain.task import PretrainTask

set_seed(0)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run training with Hydra configuration."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = TrainingConfig.from_dict(cfg_dict)

    # Initialize logging
    trackio.init(
        project=config.logging.project_name,
        auto_log_gpu=config.logging.auto_log_gpu,
        name=config.run_name,
        config=config.get_dict(),
        space_id=None,
    )

    PretrainTask(config).train()

    trackio.finish()


if __name__ == "__main__":
    main()
