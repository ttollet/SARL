# To be run on a SLURM login node.
# Based on advice from:
# - https://ax.dev/docs/0.5.0/tutorials/submitit/
# - https://ax.dev/docs/0.5.0/bayesopt/#tradeoff-between-parallelism-and-total-number-of-trials

# %% Setup
# Imports
import time
from itertools import product

import numpy as np
from submitit import AutoExecutor, LocalJob, DebugJob
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

from sarl.train import main

# Constants
# - set every time:
LOCAL_DEBUG_MODE = True  # Set to True for local debugging
SUBMITIT_DIR = "sarl/scripts/ax/submitit"
HYDRA_CONFIG_PATH = "../../config"
# - computation:
CPU_CORES_PER_TASK = 4
MAX_TRIALS = 10  # Big effect on duration
PARALLEL_LIMIT = 2 # 40
TRAIN_EPISODES = int(1000000 / 10)
# - common bounds
MIN_LR = 1e-6
MAX_LR = 1e-3
MIN_GAMMA = 0.9
MAX_GAMMA = 0.9999
# - misc:
SEED = 42
ENVS = ["platform"]  # ["platform", "goal"]
UPDATE_RATIO_PARAM = RangeParameterConfig(
    name="update_ratio",
    bounds=(0.1, 0.9),
    parameter_type="float",
    scaling="log"
)
DISCRETE_ALGS = ["ppo"]
CONTINUOUS_ALGS = ["ppo"]
# DISCRETE_ALGS = ["a2c", "dqn", "ppo"]
# CONTINUOUS_ALGS = ["a2c", "ddpg", "ppo", "sac", "td3"]


# ---- SCRIPT START ----
def get_params_by_alg(label:str = ""):
    shared_params = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=(MIN_LR, MAX_LR),
            parameter_type="float",
            scaling="log"
        ),
        RangeParameterConfig(
            name=f"{label}_gamma",
            bounds=(MIN_GAMMA, MAX_GAMMA),
            parameter_type="float",
            scaling="log"
        ),
    ]
    return {
        "a2c": shared_params + [],
        "dqn": shared_params + [],
        "ppo": shared_params + [],
        "ddpg": shared_params + [],
        "sac": shared_params + [],
        "td3": shared_params + [],
    }

pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(get_params_by_alg().keys(), repeat=2)]
cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"


# %% Optimisation
for pair, env in list(product(pairs, ENVS)):
    # STEP 1 - Define the Function to Optimise
    def objective_function(params: dict[str, list[RangeParameterConfig]]):
        # DONE: [1] Calls train.py w/o params
        # DONE: [2] Calls train.py with correct alg/env
        GlobalHydra.instance().clear()  # critical reset
        with initialize(config_path=HYDRA_CONFIG_PATH):
            cfg = compose(config_name="sarl", overrides=[
                f"hydra.job.name={pair.replace('-', '_')}-{env}-{SEED}",
                f"algorithm={pair}",
                f"environment={env}",
                f"parameters.seeds={SEED}",
                f"parameters.train_episodes={TRAIN_EPISODES}",
                # DONE: [3] Calls train.py with correct alg/env and params
                # f"params={params}"
            ])
            HydraConfig.instance().set_config(cfg)  # manually register config
            (mean_reward, std_reward) = main(cfg)  # TODO: [4] main() returns mean_reward of trained policy
        return {"mean_reward": mean_reward, "std_reward": std_reward}

    # STEP 2 - Setup Experiment via Ax
    client = Client()
    alg1, alg2 = pair.split("-")
    params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
    params = params + [UPDATE_RATIO_PARAM]
    client.configure_experiment(name="sarl_opt", parameters=params)
    client.configure_optimization(objective="mean_reward")

    # STEP 3 - Setup SubmitIt
    executor = AutoExecutor(folder=SUBMITIT_DIR, cluster=cluster)
    executor.update_parameters(timeout_min=60) # Timeout of the slurm job. Not including slurm scheduling delay.
    executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)

    # STEP 4 - Run the Experiment
    jobs = []
    submitted_jobs = 0
    while submitted_jobs < MAX_TRIALS or jobs:
        for job, trial_index in jobs[:]:
            # Monitor for completed jobs
            if job.done() or type(job) in [LocalJob, DebugJob]:
                results = job.result()
                _ = client.complete_trial(trial_index=trial_index, raw_data=results)
                _ = jobs.remove((job, trial_index))
            time.sleep(1)

        trial_index_to_param = client.get_next_trials(
            min(PARALLEL_LIMIT - len(jobs), MAX_TRIALS - submitted_jobs)
        )
        for trial_index, parameters in trial_index_to_param.items():
            job = executor.submit(objective_function, parameters)
            submitted_jobs += 1
            jobs.append((job, trial_index))
            time.sleep(1)

    time.sleep(30)
# TODO: Save best parameterisations & corresponding mean rewards
# TODO: Save visualisations
