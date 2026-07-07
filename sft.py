"""Entry point for supervised fine-tuning (SFT)."""

import hydra
import trackio
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf

from pretrain.sft import SFTRunConfig, SFTTask

set_seed(0)


@hydra.main(version_base=None, config_path="configs", config_name="sft")
def main(cfg: DictConfig) -> None:
    """Run SFT with Hydra configuration."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = SFTRunConfig(**cfg_dict)

    # trackio is auto-attached to the TRL Trainer via report_to="trackio"; we
    # only initialize/finish the run around it (mirrors main.py).
    trackio.init(
        project=config.logging.project_name,
        auto_log_gpu=config.logging.auto_log_gpu,
        name=config.run_name,
        config=config.get_dict(),
        space_id=None,
    )

    SFTTask(config).train()

    trackio.finish()


if __name__ == "__main__":
    main()
