# python3 train.py +experiment=pdqn-platform
# python3 train.py +experiment=pdqn-goal

import hydra
from hydra.utils import HydraConfig
from omegaconf import DictConfig
import warnings
import logging

# For cleaner output, mutes unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


@hydra.main(version_base=None, config_path="config", config_name="sarl")
def main(job_config: DictConfig):
    # Identify relevant script
    from snippets.temp_pdqn import test_pdqn_platform, test_pdqn_goal
    try:
        chosen_script = {
            "pdqn": {
                "platform": test_pdqn_platform,
                "goal": test_pdqn_goal
            }
        }[job_config["algorithm"]][job_config["environment"]]
    except:
        raise NotImplementedError

    # enable useful log messages, saved to /outputs
    hydra_config = HydraConfig.get()
    logger = logging.getLogger(hydra_config.job.name)
    logger.setLevel(getattr(logging, job_config.get("verbose", "info").upper()))
    logger.info("This is a proof of concept.")
    
    # Proof of concept: hydra for read/write experiments
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # See /.hydra in relevant folder for config
    return chosen_script(**job_config["parameters"])


if __name__ == "__main__":
    main()