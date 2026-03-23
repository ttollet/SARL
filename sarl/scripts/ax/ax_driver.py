# To be run on a SLURM login node.
#
# Based on advice from:
# - https://ax.dev/docs/0.5.0/tutorials/submitit/
# - https://ax.dev/docs/0.5.0/bayesopt/#tradeoff-between-parallelism-and-total-number-of-trials
# 2026-02-24
# TODO [wip] 2026-02-13 Produce a 3x3 Grid of low-med-high to demonstrate learnable parameters (fixed update ratio)
# DONE [x] Avg across n seeds
# TODO [ ] Visualise trained agents
# TODO [ ] Store best performance instead of just mean performance

# %% Setup
# TODO: Visualise the agent performance on baseline vs our pairs
# INFO: Consider that neural network size may have very large effect on performance
# TODO: Add duration to CSV
# Imports
import os
import yaml
from pathlib import Path
from pickletools import dis
from profile import run
from sqlite3.dbapi2 import paramstyle
import time
from datetime import datetime
from itertools import product
import warnings
from statistics import mean

import numpy as np
import pandas as pd
from tqdm import tqdm
from submitit import AutoExecutor, LocalJob, DebugJob
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

from sarl.train import main

warnings.filterwarnings("ignore")

# ===Constants===
# ROOT_STR = "./sarl/scripts/ax"
ROOT_STR = "."
LOCAL_DEBUG_MODE = False  # INFO: Disable for production (SLURM mode)
CPU_CORES_PER_TASK = 1  # 1 core for serial partition compatibility
HYDRA_CONFIG_PATH = "../../config"
TRAIN_EPISODES = 40_000  # WARN: Not used for converter, here for ease
# ---[on-policy algs]---
ON_POLICY_PARAMS = {"n_steps": 100}  # TODO: Vary this parameter
# ---[common bounds]---
BOUNDS_LR = (1e-6, 1e-2)
BOUNDS_UPDATE_RATIO = (0.01, 0.99)

# ---[common choices]---
# ~15 seconds per seed | platform dqn-sac
LS_TOY = 1000
CYC_TOY = 2

# 2 minutes per seed | platform dqn-sac
LS_MIN = 10_000
CYC_MIN = 8

# ??.??s per seed | platform dqn-sac
LS_PROPER = 80_000
CYC_PROPER = 16

# ---[Core settings]---
MAX_TRIALS = 1  # Big effect on duration
PARALLEL_LIMIT = 150  # 9 trials × 15 seeds = 135 parallel jobs
LEARNING_STEPS = LS_PROPER  # 80,000 steps for proper learning
CYCLES = CYC_PROPER        # 16 cycles
NUM_SEEDS = 15             # 15 seeds for variance reduction
ENVS = ["platform"]
DISCRETE_ALGS = ["dqn"]  # DQN platform, PPO goal
CONTINUOUS_ALGS = ["sac"]  # SAC best in paper
# ENVS = ["platform", "goal"]
# DISCRETE_ALGS = ["a2c", "dqn", "ppo"]
# CONTINUOUS_ALGS = ["a2c", "ddpg", "ppo", "sac", "td3"]

# 3×3 grid: use Ax's attach_trial for manual parameterizations
LR_LOW = 1e-4
LR_MED = 3.16e-3
LR_HIGH = 1e-2
GRID_PARAMS = [
    {"discrete_lr": LR_LOW,   "continuous_lr": LR_LOW,   "update_ratio": 0.5},
    {"discrete_lr": LR_MED,"continuous_lr": LR_LOW,   "update_ratio": 0.5},
    {"discrete_lr": LR_HIGH,   "continuous_lr": LR_LOW,   "update_ratio": 0.5},
    {"discrete_lr": LR_LOW,   "continuous_lr": LR_MED,"update_ratio": 0.5},
    {"discrete_lr": LR_MED,"continuous_lr": LR_MED,"update_ratio": 0.5},
    {"discrete_lr": LR_HIGH,   "continuous_lr": LR_MED,"update_ratio": 0.5},
    {"discrete_lr": LR_LOW,   "continuous_lr": LR_HIGH,   "update_ratio": 0.5},
    {"discrete_lr": LR_MED,"continuous_lr": LR_HIGH,   "update_ratio": 0.5},
    {"discrete_lr": LR_HIGH,   "continuous_lr": LR_HIGH,   "update_ratio": 0.5},
]

USE_GRID = True

NUM_SEEDS = 15  # Number of seeds for variance reduction
BASE_SEED = 1000  # Base seed for reproducibility
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]

## %% ---- SCRIPT START ----
yyyy_mm_dd_hhmm = datetime.now().strftime("%Y-%m-%d_%H-%M")
run_dir = f"{ROOT_STR}/runs/{yyyy_mm_dd_hhmm}"
Path(run_dir).mkdir(parents=True, exist_ok=True)
cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"
width = os.get_terminal_size().columns

update_ratio_param = RangeParameterConfig(
    name="update_ratio",
    bounds=BOUNDS_UPDATE_RATIO,
    parameter_type="float",
    scaling="linear"  # Linear makes sense given smaller scale
)


def create_config(run_dir):
    config = {
        'run_timestamp': yyyy_mm_dd_hhmm,
            'settings': {
                'algorithm': DISCRETE_ALGS[0] + '-' + CONTINUOUS_ALGS[0],  # or use pairs
                'environment': ENVS[0],
                'seeds': SEEDS,
                'learning_steps': LEARNING_STEPS,
                'cycles': CYCLES,
                'train_episodes': TRAIN_EPISODES,
                'on_policy_n_steps': ON_POLICY_PARAMS['n_steps']
            }
        }
    config_path = f"{Path(run_dir)}/config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
create_config(run_dir)


def get_params_by_alg(label:str = ""):
    shared_params = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=(BOUNDS_LR[0], BOUNDS_LR[1]),
            parameter_type="float",
            scaling="log"  #"linear"  # Use log for large scale
        )
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
# INFO: From source code of ax: the aqcuisition function used is qLogNoisyExpectedImprovement in BoTorch.
# See line 29 of https://github.com/facebook/Ax/blob/main/ax/generators/torch/botorch_modular/utils.py
def optimise(param_set=None, max_trials=1):
    for pair, env in list(product(pairs, ENVS)):
        def get_client():
            # Setup Experiment via Ax
            client = Client()
            alg1, alg2 = pair.split("-")
            params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
            params = params + [update_ratio_param]
            client.configure_experiment(name="sarl_opt", parameters=params)  # INFO: What to guess [A]
            client.configure_optimization(objective="mean_reward")  # INFO: What to optimise [B]
            return client
        client = get_client()

        def get_executor():
            # Setup SubmitIt
            # executor = AutoExecutor(folder="submitit", cluster=cluster)
            executor = AutoExecutor(folder=f"{run_dir}/submitit", cluster=cluster)
            executor.update_parameters(timeout_min=180) # Timeout of the slurm job. Not including slurm scheduling delay.
            executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)
            return executor
        executor = get_executor()

        # def objective_function(params: dict[str, list[RangeParameterConfig]]):
        def objective_function(params: dict[str, float], trial_index=None):
            # Define the Function to Optimise.
            # Calls main() from train.py with an automated hydra config.
            GlobalHydra.instance().clear()  # critical reset
            with initialize(config_path=HYDRA_CONFIG_PATH, job_name=(f"trial_{trial_index}-{pair.replace('-', '_')}-{env}-{SEEDS}")):
                if param_set is not None:
                    print("[DEBUG] Using fixed parameters!")
                    discrete_lr, continuous_lr, update_ratio = param_set
                else:
                    discrete_lr = params['discrete_learning_rate']
                    continuous_lr = params['continuous_learning_rate']
                    update_ratio = params['update_ratio']
                cfg = compose(config_name="sarl", return_hydra_config=True, overrides=[
                        f"algorithm={pair}",
                        f"environment={env}",
                        f"parameters.seeds={SEEDS}",
                        f"parameters.train_episodes={TRAIN_EPISODES}",
                        f"parameters.learning_steps={LEARNING_STEPS}",
                        f"parameters.cycles={CYCLES}",
                        f"parameters.alg_params.discrete_learning_rate={discrete_lr}",
                        f"parameters.alg_params.continuous_learning_rate={continuous_lr}",
                        f"parameters.alg_params.update_ratio={update_ratio}",
                        f"parameters.alg_params.on_policy_params.n_steps={ON_POLICY_PARAMS['n_steps']}",
                        f"hydra.run.dir={run_dir}/trials/trial_{trial_index}/$${{hydra.job.name}}"
                    ])
                HydraConfig.instance().set_config(cfg)  # manually register config
                mean_reward, mean_reward_se = main(cfg)
            return {"mean_reward": (mean_reward, mean_reward_se)}

        def save_client(client, wip=False):
            # Path(ROOT_STR).mkdir(parents=True, exist_ok=True)
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            if wip:
                df = client.summarize()
                # df.to_csv(f"{ROOT_STR}/wip-{yyyy_mm_dd_hhmm}-client.csv", index=False)
                df.to_csv(f"{run_dir}/wip-{yyyy_mm_dd_hhmm}-client.csv", index=False)
            else:
                if param_set is not None:
                    param_set_str = "_".join([str(param) for param in param_set])
                    client.summarize().to_csv(f"{run_dir}/{yyyy_mm_dd_hhmm}-client-{param_set_str}.csv", index=False)
                    client.save_to_json_file(f"{run_dir}/{yyyy_mm_dd_hhmm}-client-{param_set_str}.json")
                else:
                    client.summarize().to_csv(f"{run_dir}/{yyyy_mm_dd_hhmm}-client.csv", index=False)
                    client.save_to_json_file(f"{run_dir}/{yyyy_mm_dd_hhmm}-client.json")
            return True

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
            # results = []
            while submitted_jobs < max_trials or jobs:
                def run_trials():
                    global submitted_jobs
                    trial_index_to_param = client.get_next_trials(  # INFO: Ax makes guesses [C]
                        min(PARALLEL_LIMIT - len(jobs), max_trials - submitted_jobs)
                    )
                    for trial_index, parameters in trial_index_to_param.items():
                        print(f"[INFO] Submitting parameters: {parameters}")
                        job = executor.submit(objective_function, parameters, trial_index)  # TODO: Store job ID and params
                        submitted_jobs += 1
                        jobs.append((job, trial_index))
                        time.sleep(1)
                def learn_from_any_previous_trials():
                    for job, trial_index in jobs[:]:  # INFO: Ax learns how any previous guesses went [D]
                        # Monitor for completed jobs
                        if job.done() or type(job) in [LocalJob, DebugJob]:
                            result = job.result()  # TODO: Append variance to result
                            # mean_reward = result['mean_reward']
                            print(f"\n[JOB RESULT]: {result}")
                            print("-" * width)
                            _ = client.complete_trial(trial_index=trial_index, raw_data=result)
                            save_client(client, wip=True)
                            # results.append((mean_reward, hyperparameters))
                            _ = jobs.remove((job, trial_index))
                        # WARN: Reintroduce sleep() for Slurm
                        # time.sleep(1)
                run_trials()
                learn_from_any_previous_trials()
                # WARN: Reintroduce sleep() for Slurm
                # time.sleep(30)
            best = client.get_best_parameterization()  # TODO: Save best parameterisations & corresponding mean rewards
            save_client(client)
            # best_param, best_mean_reward = 0, 0
            return {"best": best}
        outcome = run_parallel_exps()
        print(f"\n[RESULT] {outcome['best'][0]} results in {outcome['best'][1]} observed on trial {outcome['best'][2]}")
        print("-" * width)


def run_single_seed(grid_params, trial_index, seed):
    """Run a single seed with grid parameters. Each seed is a separate SLURM job."""
    discrete_lr = grid_params["discrete_lr"]
    continuous_lr = grid_params["continuous_lr"]
    update_ratio = grid_params["update_ratio"]

    # Folder name: trial_0_d1e-4_c1e-4_u0.5/seed_12345
    job_name = f"trial_{trial_index}_d{discrete_lr:.0e}_c{continuous_lr:.0e}_u{update_ratio}_seed{seed}"

    GlobalHydra.instance().clear()
    with initialize(config_path=HYDRA_CONFIG_PATH, job_name=job_name):
        cfg = compose(config_name="sarl", return_hydra_config=True, overrides=[
            f"algorithm={DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}",
            f"environment={ENVS[0]}",
            f"parameters.seeds=[{seed}]",  # Single seed
            f"parameters.train_episodes={TRAIN_EPISODES}",
            f"parameters.learning_steps={LEARNING_STEPS}",
            f"parameters.cycles={CYCLES}",
            f"parameters.alg_params.discrete_learning_rate={discrete_lr}",
            f"parameters.alg_params.continuous_learning_rate={continuous_lr}",
            f"parameters.alg_params.update_ratio={update_ratio}",
            f"parameters.alg_params.on_policy_params.n_steps={ON_POLICY_PARAMS['n_steps']}",
            f"hydra.run.dir={run_dir}/trials/trial_{trial_index}_d{discrete_lr:.0e}_c{continuous_lr:.0e}_u{update_ratio}/seed_{seed}/$${{hydra.job.name}}"
        ])
        HydraConfig.instance().set_config(cfg)
        mean_reward, mean_reward_se = main(cfg)

    return {"mean_reward": mean_reward, "mean_reward_se": mean_reward_se}


def run_trial_from_grid(grid_params, trial_index):
    """Run a single trial with grid parameters (all seeds sequentially - for debug mode)."""
    discrete_lr = grid_params["discrete_lr"]
    continuous_lr = grid_params["continuous_lr"]
    update_ratio = grid_params["update_ratio"]

    job_name = f"trial_{trial_index}_d{discrete_lr:.0e}_c{continuous_lr:.0e}"

    GlobalHydra.instance().clear()
    with initialize(config_path=HYDRA_CONFIG_PATH, job_name=job_name):
        cfg = compose(config_name="sarl", return_hydra_config=True, overrides=[
            f"algorithm={DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}",
            f"environment={ENVS[0]}",
            f"parameters.seeds={SEEDS}",
            f"parameters.train_episodes={TRAIN_EPISODES}",
            f"parameters.learning_steps={LEARNING_STEPS}",
            f"parameters.cycles={CYCLES}",
            f"parameters.alg_params.discrete_learning_rate={discrete_lr}",
            f"parameters.alg_params.continuous_learning_rate={continuous_lr}",
            f"parameters.alg_params.update_ratio={update_ratio}",
            f"parameters.alg_params.on_policy_params.n_steps={ON_POLICY_PARAMS['n_steps']}",
            f"hydra.run.dir={run_dir}/trials/trial_{trial_index}_d{discrete_lr:.0e}_c{continuous_lr:.0e}_u{update_ratio}/$${{hydra.job.name}}"
        ])
        HydraConfig.instance().set_config(cfg)
        mean_reward, mean_reward_se = main(cfg)

    return {"mean_reward": (mean_reward, mean_reward_se)}


def run_grid_search(client):
    """Run grid search with ALL seeds as separate SLURM jobs (full parallelism)."""
    all_results = []

    # Setup executor for parallel SLURM jobs
    executor = AutoExecutor(folder=f"{run_dir}/submitit", cluster=cluster)
    executor.update_parameters(timeout_min=60)  # ~10 min per seed
    executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)

    # Determine if we should run in parallel or sequential mode
    run_in_parallel = not LOCAL_DEBUG_MODE

    if run_in_parallel:
        print(f"[INFO] Running in FULL PARALLEL mode: {len(GRID_PARAMS)} trials × {len(SEEDS)} seeds = {len(GRID_PARAMS) * len(SEEDS)} parallel jobs")
    else:
        print(f"[INFO] Running in SEQUENTIAL mode (debug): {len(GRID_PARAMS)} trials × {len(SEEDS)} seeds")

    # Attach all trials to Ax first
    trial_indices = {}
    for grid_params in GRID_PARAMS:
        trial_index = client.attach_trial(
            parameters={
                "discrete_learning_rate": grid_params["discrete_lr"],
                "continuous_learning_rate": grid_params["continuous_lr"],
                "update_ratio": grid_params["update_ratio"],
            },
            arm_name=f"grid_d{grid_params['discrete_lr']:.0e}_c{grid_params['continuous_lr']:.0e}_u{grid_params['update_ratio']}"
        )
        trial_indices[id(grid_params)] = trial_index

    if run_in_parallel:
        # Submit ALL 135 jobs at once
        print(f"[INFO] Submitting all {len(GRID_PARAMS) * len(SEEDS)} jobs to SLURM...")

        all_jobs = []
        for grid_params in GRID_PARAMS:
            trial_index = trial_indices[id(grid_params)]
            for seed in SEEDS:
                job = executor.submit(run_single_seed, grid_params, trial_index, seed)
                all_jobs.append({
                    'job': job,
                    'trial_index': trial_index,
                    'grid_params': grid_params,
                    'seed': seed
                })

        print(f"[INFO] All jobs submitted. Waiting for completion...")

        # Collect results by trial
        trial_results = {}
        for job_info in all_jobs:
            job = job_info['job']
            trial_index = job_info['trial_index']
            grid_params = job_info['grid_params']
            seed = job_info['seed']

            print(f"[INFO] Waiting for trial {trial_index}, seed {seed}...")
            result = job.result()

            if trial_index not in trial_results:
                trial_results[trial_index] = {'grid_params': grid_params, 'rewards': []}
            trial_results[trial_index]['rewards'].append(result['mean_reward'])
            print(f"[INFO] Trial {trial_index}, seed {seed} completed: mean_reward = {result['mean_reward']:.4f}")

        # Compute stats for each trial and report to Ax
        for trial_index, data in trial_results.items():
            rewards = np.array(data['rewards'])
            mean_reward = float(np.mean(rewards))
            sem = float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))

            # Report results to Ax with SE
            client.complete_trial(trial_index=trial_index, raw_data={"mean_reward": (mean_reward, sem)})

            grid_params = data['grid_params']
            all_results.append({
                **grid_params,
                'trial_index': trial_index,
                'num_seeds': len(SEEDS),
                'mean_reward': mean_reward,
                'std_error': sem
            })

            print(f"[INFO] Trial {trial_index} complete: mean_reward = {mean_reward:.4f} ± {sem:.4f}")

    else:
        # Sequential mode (debug) - run all seeds in one job
        for grid_params in GRID_PARAMS:
            trial_index = trial_indices[id(grid_params)]
            result = run_trial_from_grid(grid_params, trial_index)
            mean_reward = result['mean_reward'][0]
            sem = result['mean_reward'][1]

            # Report results to Ax
            client.complete_trial(trial_index=trial_index, raw_data=result)

            all_results.append({
                **grid_params,
                'trial_index': trial_index,
                'num_seeds': len(SEEDS),
                'mean_reward': mean_reward,
                'std_error': sem
            })

            # Save progress after each trial
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(f"{run_dir}/wip-grid-results.csv", index=False)
            print(f"[INFO] Trial {trial_index} complete: mean_reward = {mean_reward:.4f} ± {sem:.4f}")

    # Save final results with SE
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{run_dir}/wip-grid-results.csv", index=False)
    results_df.to_csv(f"{run_dir}/grid_results.csv", index=False)
    print(f"[INFO] Saved {len(results_df)} results to {run_dir}/grid_results.csv")

    return results_df


if USE_GRID:
    # Setup client for grid search (Ax best practice for manual trials)
    pair = f"{DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}"
    client = Client()
    alg1, alg2 = pair.split("-")
    params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
    params = params + [update_ratio_param]
    client.configure_experiment(name="sarl_grid", parameters=params)
    client.configure_optimization(objective="mean_reward")

    results_df = run_grid_search(client)
    print(f"[INFO] Grid search complete!")
else:
    optimise(max_trials=MAX_TRIALS)
