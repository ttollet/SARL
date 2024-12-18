# python3 sarl/train.py +experiment=pdqn-platform
# python3 sarl/train.py +experiment=pdqn-goal
# python3 sarl/train.py +experiment=pdqn-platform parameters.train_episodes=50000
# python3 sarl/train.py +experiment=ppo-ppo-platform-1h-15seeds parameters.seeds=[1,2]

# tensorboard --logdir=outputs

from functools import partial
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import warnings
import logging

# For cleaner output, mutes unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


@hydra.main(version_base=None, config_path="config", config_name="sarl")
def main(job_config: DictConfig):
    # Identify relevant script
    from sarl.snippets.temp_pdqn import test_pdqn_platform, test_pdqn_goal  # TODO: Refactor to avoid script-per-combination
    from sarl.snippets.converter_use import run_converter  # TODO: Use this via the partial method
    from sarl.snippets.converter_use import ppo_ppo_platform, a2c_ppo_platform, dqn_ppo_platform, ppo_ppo_goal, a2c_ppo_goal, dqn_ppo_goal
    from sarl.snippets.converter_use import ppo_a2c_platform, a2c_a2c_platform, dqn_a2c_platform, ppo_a2c_goal, a2c_a2c_goal, dqn_a2c_goal
    from sarl.snippets.converter_use import ppo_ddpg_platform, a2c_ddpg_platform, dqn_ddpg_platform, ppo_ddpg_goal, a2c_ddpg_goal, dqn_ddpg_goal
    from sarl.snippets.converter_use import ppo_sac_platform, a2c_sac_platform, dqn_sac_platform, ppo_sac_goal, a2c_sac_goal, dqn_sac_goal
    from sarl.snippets.converter_use import ppo_td3_platform, a2c_td3_platform, dqn_td3_platform, ppo_td3_goal, a2c_td3_goal, dqn_td3_goal
    try:
        chosen_script = {  # (Dict mapping config terms to functions)
            "pdqn": {
                "platform": test_pdqn_platform,
                "goal": test_pdqn_goal
            },

            "ppo-ppo": {
                "platform": ppo_ppo_platform,
                "goal": ppo_ppo_goal
            },
            "a2c-ppo": {
                "platform": a2c_ppo_platform,
                "goal": a2c_ppo_goal
            },
            "dqn-ppo": {
                "platform": dqn_ppo_platform,
                "goal": dqn_ppo_goal
            },

            "ppo-a2c": {
                "platform": ppo_a2c_platform,
                "goal": ppo_a2c_goal
            },
            "a2c-a2c": {
                "platform": a2c_a2c_platform,
                "goal": a2c_a2c_goal
            },
            "dqn-a2c": {
                "platform": dqn_a2c_platform,
                "goal": dqn_a2c_goal
            },

            "ppo-ddpg": {
                "platform": ppo_ddpg_platform,
                "goal": ppo_ddpg_goal
            },
            "a2c-ddpg": {
                "platform": a2c_ddpg_platform,
                "goal": a2c_ddpg_goal
            },
            "dqn-ddpg": {
                "platform": dqn_ddpg_platform,
                "goal": dqn_ddpg_goal
            },

            "ppo-sac": {
                "platform": ppo_sac_platform,
                "goal": ppo_sac_goal
            },
            "a2c-sac": {
                "platform": a2c_sac_platform,
                "goal": a2c_sac_goal
            },
            "dqn-sac": {
                "platform": dqn_sac_platform,
                "goal": dqn_sac_goal
            },

            "ppo-td3": {
                "platform": ppo_td3_platform,
                "goal": ppo_td3_goal
            },
            "a2c-td3": {
                "platform": a2c_td3_platform,
                "goal": a2c_td3_goal
            },
            "dqn-td3": {
                "platform": dqn_td3_platform,
                "goal": dqn_td3_goal
            },
        }[job_config["algorithm"]][job_config["environment"]]
    except:
        raise NotImplementedError

    # enable useful log messages, saved to /outputs
    hydra_config = HydraConfig.get()
    logger = logging.getLogger(hydra_config.job.name)
    logger.setLevel(getattr(logging, job_config.get("verbose", "info").upper()))
    logger.info("This is a proof of concept.")

    # Proof of concept: hydra for read/write experiments
    output_dir = HydraConfig.get().runtime.output_dir  # See /.hydra in relevant folder for config
    return chosen_script(**job_config["parameters"], output_dir=output_dir)


if __name__ == "__main__":
    main()
