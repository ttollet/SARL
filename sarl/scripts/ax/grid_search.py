"""
Grid search execution functions.
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd
from submitit import AutoExecutor

from config import (
    GRID_PARAMS, SEEDS, LOCAL_DEBUG_MODE,
    CPU_CORES_PER_TASK, run_dir, cluster,
    BASE_SEED, NUM_SEEDS, ROTATE_SEEDS_PER_TRIALS
)
from training import run_training


def format_duration(seconds):
    """Format seconds as hours:minutes:seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


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


def save_grid_timing_summary(total_duration, trial_durations, num_trials, num_seeds):
    """Save timing summary to CSV."""
    timing_data = {
        'total_duration_seconds': total_duration,
        'total_duration': format_duration(total_duration),
        'num_trials': num_trials,
        'num_seeds_per_trial': num_seeds,
        'total_jobs': num_trials * num_seeds,
        'avg_job_duration_seconds': np.mean(trial_durations) if trial_durations else None,
        'avg_job_duration': format_duration(np.mean(trial_durations)) if trial_durations else None,
        'min_job_duration_seconds': np.min(trial_durations) if trial_durations else None,
        'max_job_duration_seconds': np.max(trial_durations) if trial_durations else None,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    timing_df = pd.DataFrame([timing_data])
    timing_df.to_csv(f"{run_dir}/timing_summary.csv", index=False)
    print(f"[INFO] Saved timing summary to {run_dir}/timing_summary.csv")


def run_grid_search(client, learning_steps=None, cycles=None, seeds=None):
    """Run grid search with ALL seeds as separate SLURM jobs (full parallelism)."""
    if seeds is None:
        seeds = SEEDS
    if learning_steps is None:
        learning_steps = 80_000
    if cycles is None:
        cycles = 16

    run_start_time = time.time()
    all_results = []
    all_seed_results = []
    all_job_durations = []

    executor = AutoExecutor(folder=f"{run_dir}/submitit", cluster=cluster)
    executor.update_parameters(timeout_min=60)
    executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)

    run_in_parallel = not LOCAL_DEBUG_MODE

    print(f"[INFO] Grid search started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if run_in_parallel:
        print(f"[INFO] Running in FULL PARALLEL mode: {len(GRID_PARAMS)} trials × {len(seeds)} seeds = {len(GRID_PARAMS) * len(seeds)} parallel jobs")
    else:
        print(f"[INFO] Running in SEQUENTIAL mode (debug): {len(GRID_PARAMS)} trials × {len(seeds)} seeds")

    if ROTATE_SEEDS_PER_TRIALS:
        print(f"[INFO] ROTATE_SEEDS_PER_TRIALS=True: Each trial will use different seeds (sequential chunks)")

    trial_indices = {}
    for grid_idx, grid_params in enumerate(GRID_PARAMS):
        # Compute per-trial seeds if rotation is enabled
        if ROTATE_SEEDS_PER_TRIALS:
            trial_seeds = [BASE_SEED + grid_idx * NUM_SEEDS + i for i in range(NUM_SEEDS)]
        else:
            trial_seeds = seeds

        trial_index = client.attach_trial(
            parameters={
                "discrete_learning_rate": grid_params["discrete_lr"],
                "continuous_learning_rate": grid_params["continuous_lr"],
                "update_ratio": grid_params["update_ratio"],
            },
            arm_name=f"grid_d{grid_params['discrete_lr']:.0e}_c{grid_params['continuous_lr']:.0e}_u{grid_params['update_ratio']}"
        )
        trial_indices[id(grid_params)] = (trial_index, trial_seeds)

    if run_in_parallel:
        print(f"[INFO] Submitting all {len(GRID_PARAMS) * len(seeds)} jobs to SLURM...")

        all_jobs = []
        for grid_params in GRID_PARAMS:
            trial_index, trial_seeds = trial_indices[id(grid_params)]
            for seed in trial_seeds:
                job = executor.submit(run_single_seed, grid_params, trial_index, seed, learning_steps, cycles)
                all_jobs.append({
                    'job': job,
                    'trial_index': trial_index,
                    'grid_params': grid_params,
                    'seed': seed,
                    'start_time': time.time()
                })

        print(f"[INFO] All jobs submitted. Waiting for completion...")

        trial_results = {}
        for job_info in all_jobs:
            job = job_info['job']
            trial_index = job_info['trial_index']
            grid_params = job_info['grid_params']
            seed = job_info['seed']
            start_time = job_info['start_time']

            print(f"[INFO] Waiting for trial {trial_index}, seed {seed}...")
            result = job.result()
            job_duration = time.time() - start_time
            all_job_durations.append(job_duration)

            # Save per-seed result with duration
            all_seed_results.append({
                'trial_index': trial_index,
                'discrete_lr': grid_params['discrete_lr'],
                'continuous_lr': grid_params['continuous_lr'],
                'update_ratio': grid_params['update_ratio'],
                'seed': seed,
                'mean_reward': result['mean_reward'],
                'duration_seconds': job_duration,
                'duration': format_duration(job_duration)
            })

            if trial_index not in trial_results:
                trial_results[trial_index] = {'grid_params': grid_params, 'rewards': []}
            trial_results[trial_index]['rewards'].append(result['mean_reward'])
            print(f"[INFO] Trial {trial_index}, seed {seed} completed: mean_reward = {result['mean_reward']:.4f} ({format_duration(job_duration)})")

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
            trial_index, trial_seeds = trial_indices[id(grid_params)]
            trial_start = time.time()
            result = run_trial_from_grid(grid_params, trial_index, trial_seeds, learning_steps, cycles)
            trial_duration = time.time() - trial_start
            all_job_durations.append(trial_duration)
            mean_reward = result['mean_reward'][0]
            sem = result['mean_reward'][1]

            # Save per-seed results with duration
            all_seed_results.append({
                'trial_index': trial_index,
                'discrete_lr': grid_params['discrete_lr'],
                'continuous_lr': grid_params['continuous_lr'],
                'update_ratio': grid_params['update_ratio'],
                'seed': trial_seeds[0],
                'mean_reward': mean_reward,
                'duration_seconds': trial_duration,
                'duration': format_duration(trial_duration)
            })

            client.complete_trial(trial_index=trial_index, raw_data=result)

            all_results.append({
                **grid_params,
                'trial_index': trial_index,
                'num_seeds': len(seeds),
                'mean_reward': mean_reward,
                'std_error': sem,
                'duration_seconds': trial_duration,
                'duration': format_duration(trial_duration)
            })

            results_df = pd.DataFrame(all_results)
            results_df.to_csv(f"{run_dir}/wip-grid-results.csv", index=False)
            print(f"[INFO] Trial {trial_index} complete: mean_reward = {mean_reward:.4f} ± {sem:.4f} ({format_duration(trial_duration)})")

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

    # Save timing summary
    total_duration = time.time() - run_start_time
    print(f"\n[TIMING] Grid search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[TIMING] Total duration: {format_duration(total_duration)}")
    save_grid_timing_summary(total_duration, all_job_durations, len(GRID_PARAMS), len(seeds))

    return results_df
