"""
Grid search execution functions.
"""
import numpy as np
import pandas as pd
from submitit import AutoExecutor

from config import (
    GRID_PARAMS, SEEDS, LOCAL_DEBUG_MODE,
    CPU_CORES_PER_TASK, run_dir, cluster
)
from training import run_training


def run_single_seed(grid_params, trial_index, seed, learning_steps=None, cycles=None):
    """Run a single seed with grid parameters. Each seed is a separate SLURM job."""
    discrete_lr = grid_params["discrete_lr"]
    continuous_lr = grid_params["continuous_lr"]
    update_ratio = grid_params["update_ratio"]

    job_name = f"trial_{trial_index}_d{discrete_lr:.0e}_c{continuous_lr:.0e}_u{update_ratio}_seed{seed}"
    run_subdir = f"trial_{trial_index}_d{discrete_lr:.0e}_c{continuous_lr:.0e}_u{update_ratio}/seed_{seed}"

    mean_reward, mean_reward_se = run_training(
        discrete_lr, continuous_lr, update_ratio, [seed], job_name, run_subdir,
        learning_steps=learning_steps, cycles=cycles
    )

    return {"mean_reward": mean_reward, "mean_reward_se": mean_reward_se}


def run_trial_from_grid(grid_params, trial_index, seeds, learning_steps=None, cycles=None):
    """Run a single trial with grid parameters (all seeds sequentially - for debug mode)."""
    discrete_lr = grid_params["discrete_lr"]
    continuous_lr = grid_params["continuous_lr"]
    update_ratio = grid_params["update_ratio"]

    job_name = f"trial_{trial_index}_d{discrete_lr:.0e}_c{continuous_lr:.0e}"
    run_subdir = f"trial_{trial_index}_d{discrete_lr:.0e}_c{continuous_lr:.0e}_u{update_ratio}"

    mean_reward, mean_reward_se = run_training(
        discrete_lr, continuous_lr, update_ratio, seeds, job_name, run_subdir,
        learning_steps=learning_steps, cycles=cycles
    )

    return {"mean_reward": (mean_reward, mean_reward_se)}


def run_grid_search(client, learning_steps=None, cycles=None, seeds=None):
    """Run grid search with ALL seeds as separate SLURM jobs (full parallelism)."""
    if seeds is None:
        seeds = SEEDS
    if learning_steps is None:
        learning_steps = 80_000
    if cycles is None:
        cycles = 16

    all_results = []
    all_seed_results = []

    executor = AutoExecutor(folder=f"{run_dir}/submitit", cluster=cluster)
    executor.update_parameters(timeout_min=60)
    executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)

    run_in_parallel = not LOCAL_DEBUG_MODE

    if run_in_parallel:
        print(f"[INFO] Running in FULL PARALLEL mode: {len(GRID_PARAMS)} trials × {len(seeds)} seeds = {len(GRID_PARAMS) * len(seeds)} parallel jobs")
    else:
        print(f"[INFO] Running in SEQUENTIAL mode (debug): {len(GRID_PARAMS)} trials × {len(seeds)} seeds")

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
        print(f"[INFO] Submitting all {len(GRID_PARAMS) * len(seeds)} jobs to SLURM...")

        all_jobs = []
        for grid_params in GRID_PARAMS:
            trial_index = trial_indices[id(grid_params)]
            for seed in seeds:
                job = executor.submit(run_single_seed, grid_params, trial_index, seed, learning_steps, cycles)
                all_jobs.append({
                    'job': job,
                    'trial_index': trial_index,
                    'grid_params': grid_params,
                    'seed': seed
                })

        print(f"[INFO] All jobs submitted. Waiting for completion...")

        trial_results = {}
        for job_info in all_jobs:
            job = job_info['job']
            trial_index = job_info['trial_index']
            grid_params = job_info['grid_params']
            seed = job_info['seed']

            print(f"[INFO] Waiting for trial {trial_index}, seed {seed}...")
            result = job.result()

            # Save per-seed result
            all_seed_results.append({
                'trial_index': trial_index,
                'discrete_lr': grid_params['discrete_lr'],
                'continuous_lr': grid_params['continuous_lr'],
                'update_ratio': grid_params['update_ratio'],
                'seed': seed,
                'mean_reward': result['mean_reward']
            })

            if trial_index not in trial_results:
                trial_results[trial_index] = {'grid_params': grid_params, 'rewards': []}
            trial_results[trial_index]['rewards'].append(result['mean_reward'])
            print(f"[INFO] Trial {trial_index}, seed {seed} completed: mean_reward = {result['mean_reward']:.4f}")

        for trial_index, data in trial_results.items():
            rewards = np.array(data['rewards'])
            mean_reward = float(np.mean(rewards))
            sem = float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))

            client.complete_trial(trial_index=trial_index, raw_data={"mean_reward": (mean_reward, sem)})

            grid_params = data['grid_params']
            all_results.append({
                **grid_params,
                'trial_index': trial_index,
                'num_seeds': len(seeds),
                'mean_reward': mean_reward,
                'std_error': sem
            })

            print(f"[INFO] Trial {trial_index} complete: mean_reward = {mean_reward:.4f} ± {sem:.4f}")

    else:
        for grid_params in GRID_PARAMS:
            trial_index = trial_indices[id(grid_params)]
            result = run_trial_from_grid(grid_params, trial_index, seeds, learning_steps, cycles)
            mean_reward = result['mean_reward'][0]
            sem = result['mean_reward'][1]

            # Save per-seed results (single aggregated result from run_trial_from_grid)
            all_seed_results.append({
                'trial_index': trial_index,
                'discrete_lr': grid_params['discrete_lr'],
                'continuous_lr': grid_params['continuous_lr'],
                'update_ratio': grid_params['update_ratio'],
                'seed': seeds[0],  # Note: aggregated over seeds
                'mean_reward': mean_reward
            })

            client.complete_trial(trial_index=trial_index, raw_data=result)

            all_results.append({
                **grid_params,
                'trial_index': trial_index,
                'num_seeds': len(seeds),
                'mean_reward': mean_reward,
                'std_error': sem
            })

            results_df = pd.DataFrame(all_results)
            results_df.to_csv(f"{run_dir}/wip-grid-results.csv", index=False)
            print(f"[INFO] Trial {trial_index} complete: mean_reward = {mean_reward:.4f} ± {sem:.4f}")

    # Save aggregated results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{run_dir}/wip-grid-results.csv", index=False)
    results_df.to_csv(f"{run_dir}/grid_results.csv", index=False)
    print(f"[INFO] Saved {len(results_df)} results to {run_dir}/grid_results.csv")

    # Save per-seed results
    if all_seed_results:
        seed_df = pd.DataFrame(all_seed_results)
        seed_df.to_csv(f"{run_dir}/seed_results.csv", index=False)
        print(f"[INFO] Saved {len(seed_df)} seed results to {run_dir}/seed_results.csv")

    return results_df
