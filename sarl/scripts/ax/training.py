"""
Core training functions shared by all execution modes.
"""
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra

from sarl.train import main

from config import (
    DISCRETE_ALGS, CONTINUOUS_ALGS, ENVS,
    TRAIN_EPISODES, LEARNING_STEPS, CYCLES,
    ON_POLICY_PARAMS, HYDRA_CONFIG_PATH, run_dir
)


def build_config(discrete_lr, continuous_lr, update_ratio, seeds, job_name, run_subdir=""):
    """Build Hydra config for training. Shared by all execution modes."""
    GlobalHydra.instance().clear()
    with initialize(config_path=HYDRA_CONFIG_PATH, job_name=job_name):
        cfg = compose(config_name="sarl", return_hydra_config=True, overrides=[
            f"algorithm={DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}",
            f"environment={ENVS[0]}",
            f"parameters.seeds={seeds}",
            f"parameters.train_episodes={TRAIN_EPISODES}",
            f"parameters.learning_steps={LEARNING_STEPS}",
            f"parameters.cycles={CYCLES}",
            f"parameters.alg_params.discrete_learning_rate={discrete_lr}",
            f"parameters.alg_params.continuous_learning_rate={continuous_lr}",
            f"parameters.alg_params.update_ratio={update_ratio}",
            f"parameters.alg_params.on_policy_params.n_steps={ON_POLICY_PARAMS['n_steps']}",
            f"hydra.run.dir={run_dir}/trials/{run_subdir}/$${{hydra.job.name}}"
        ])
        HydraConfig.instance().set_config(cfg)
    return cfg


def run_training(discrete_lr, continuous_lr, update_ratio, seeds, job_name, run_subdir=""):
    """Execute training with given parameters. Returns (mean_reward, mean_reward_se)."""
    cfg = build_config(discrete_lr, continuous_lr, update_ratio, seeds, job_name, run_subdir)
    mean_reward, mean_reward_se = main(cfg)
    return mean_reward, mean_reward_se
