# To be run on a SLURM login node.
# Based on advice from:
# - https://ax.dev/docs/0.5.0/tutorials/submitit/
# - https://ax.dev/docs/0.5.0/bayesopt/#tradeoff-between-parallelism-and-total-number-of-trials

# %% Setup
# Imports
from profile import run
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
SUBMITIT_DIR = "submitit"
HYDRA_CONFIG_PATH = "../../config"
# - computation:
CPU_CORES_PER_TASK = 4
MAX_TRIALS = 40  # Big effect on duration
PARALLEL_LIMIT = 2 # 40
TRAIN_EPISODES = int(1000000 / 10)
# - common bounds
MIN_LR = 0 #1e-6
MAX_LR = 1 #1e-3
MIN_GAMMA = 0.9
MAX_GAMMA = 0.9999
# - misc:
SEED = 42
ENVS = ["platform"]  # TODO: Set correct final ENVS
# ["platform", "goal"]
UPDATE_RATIO_PARAM = RangeParameterConfig(
    name="update_ratio",
    bounds=(0.1, 0.9),
    parameter_type="float",
    scaling="linear"
)
# TODO: Set correct final *_ALGS
DISCRETE_ALGS = ["ppo"]
CONTINUOUS_ALGS = ["ppo"]  # TODO: Vary discrete alg choice first
# DISCRETE_ALGS = ["a2c", "dqn", "ppo"]
# CONTINUOUS_ALGS = ["a2c", "ddpg", "ppo", "sac", "td3"]


# ---- SCRIPT START ----
cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"

def get_params_by_alg(label:str = ""):
    shared_params = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=(MIN_LR, MAX_LR),
            parameter_type="float",
            scaling="linear"
        ),
    ]
    return {
        "a2c": shared_params + [],
        "dqn": shared_params + [],
        "ppo": shared_params + [],  # INFO: Consider batch size
        "ddpg": shared_params + [],
        "sac": shared_params + [],
        "td3": shared_params + [],
    }
# pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(get_params_by_alg().keys(), repeat=2)]
pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(DISCRETE_ALGS, CONTINUOUS_ALGS)]



# %% Optimisation
def optimise():
    for pair, env in list(product(pairs, ENVS)):
        def get_client():
            # Setup Experiment via Ax
            client = Client()
            alg1, alg2 = pair.split("-")
            params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
            params = params + [UPDATE_RATIO_PARAM]
            client.configure_experiment(name="sarl_opt", parameters=params)  # INFO: What to guess [A]
            client.configure_optimization(objective="mean_reward")  # INFO: What to optimise [B]
            return client
        client = get_client()

        def get_executor():
            # Setup SubmitIt
            executor = AutoExecutor(folder=SUBMITIT_DIR, cluster=cluster)
            executor.update_parameters(timeout_min=60) # Timeout of the slurm job. Not including slurm scheduling delay.
            executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)
            return executor
        executor = get_executor()

        # def objective_function(params: dict[str, list[RangeParameterConfig]]):
        def objective_function(params: dict[str, float]):
            # Define the Function to Optimise.
            # Calls main() from train.py with an automated hydra config.
            GlobalHydra.instance().clear()  # critical reset
            with initialize(config_path=HYDRA_CONFIG_PATH, job_name=(f"{pair.replace('-', '_')}-{env}-{SEED}")):
                cfg = compose(config_name="sarl", return_hydra_config=True, overrides=[
                        f"algorithm={pair}",
                        f"environment={env}",
                        f"parameters.seeds={SEED}",
                        f"parameters.train_episodes={TRAIN_EPISODES}",
                        f"parameters.alg_params.discrete_learning_rate={params['discrete_learning_rate']}",
                        f"parameters.alg_params.continuous_learning_rate={params['continuous_learning_rate']}",
                    ])
                HydraConfig.instance().set_config(cfg)  # manually register config
                mean_reward = main(cfg)  # TODO: [1] main() returns mean_reward
            return {"mean_reward": mean_reward}#, "std_reward": std_reward}

        # This demonstrates the objective_function works
        # def toy_func():
        #     params = {"learning_rate": 0.5}
        #     print(objective_function(params))
        #     return True
        # toy_func()

        def run_parallel_exps():
            # Run the Experiment
            """
            Returns list of:
                - The parameters predicted to have the best optimization value without
                    violating any outcome constraints.
                - The metric values for the best parameterization. Uses model prediction if
                    ``use_model_predictions=True``, otherwise returns observed data.
                - The trial which most recently ran the best parameterization
                - The name of the best arm (each trial has a unique name associated with
                    each parameterization)
            """
            jobs = []
            global submitted_jobs
            submitted_jobs = 0
            while submitted_jobs < MAX_TRIALS or jobs:
                def run_trials():
                    global submitted_jobs
                    trial_index_to_param = client.get_next_trials(  # INFO: Ax makes guesses [C]
                        min(PARALLEL_LIMIT - len(jobs), MAX_TRIALS - submitted_jobs)
                    )
                    for trial_index, parameters in trial_index_to_param.items():
                        job = executor.submit(objective_function, parameters)
                        submitted_jobs += 1
                        jobs.append((job, trial_index))
                        time.sleep(1)
                def learn_from_any_previous_trials():
                    for job, trial_index in jobs[:]:  # INFO: Ax learns how any previous guesses went [D]
                        # Monitor for completed jobs
                        if job.done() or type(job) in [LocalJob, DebugJob]:
                            results = job.result()

                            print(results)  # TODO: Check working well
                            _ = client.complete_trial(trial_index=trial_index, raw_data=results)
                            _ = jobs.remove((job, trial_index))
                        # WARN: Reintroduce sleep() for Slurm
                        # time.sleep(1)
                run_trials()
                learn_from_any_previous_trials()
                # WARN: Reintroduce sleep() for Slurm
                # time.sleep(30)
            best = client.get_best_parameterization()  # TODO: Save best parameterisations & corresponding mean rewards
            # best_param, best_mean_reward = 0, 0
            return best
        outcome = run_parallel_exps()
        print(f"\n[RESULT] {outcome[0]} results in {outcome[1]} observed on trial {outcome[2]}")

        def visualise():  # TODO: Save visualisations
            pass
optimise()
