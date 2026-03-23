"""
Bayesian optimization functions using Ax.
"""
import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from submitit import AutoExecutor, LocalJob, DebugJob

from config import (
    pairs, ENVS, SEEDS,
    MAX_TRIALS, PARALLEL_LIMIT,
    update_ratio_param, get_params_by_alg,
    CPU_CORES_PER_TASK, run_dir, cluster
)
from training import run_training


best_scores_history = []
best_so_far = None
run_start_time = None


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


def track_best_score(mean_reward, trial_index, params, duration=None):
    """Track best observed score over time."""
    global best_so_far
    if best_so_far is None or mean_reward > best_so_far:
        best_so_far = mean_reward
    entry = {
        'trial': trial_index,
        'mean_reward': mean_reward,
        'best_so_far': best_so_far,
        'params': params
    }
    if duration is not None:
        entry['duration_seconds'] = duration
        entry['duration'] = format_duration(duration)
    best_scores_history.append(entry)


def save_client(client, wip=False):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    if wip:
        df = client.summarize()
        df.to_csv(f"{run_dir}/wip-client.csv", index=False)
    else:
        df = client.summarize()
        df.to_csv(f"{run_dir}/client.csv", index=False)
        client.save_to_json_file(f"{run_dir}/client.json")
    return True


def plot_best_scores(output_dir=None):
    """Plot best observed scores from BO run."""
    if output_dir is None:
        output_dir = run_dir

    if not best_scores_history:
        print("[WARN] No best scores history to plot")
        return

    trials = [h['trial'] for h in best_scores_history]
    rewards = [h['mean_reward'] for h in best_scores_history]
    best = [h['best_so_far'] for h in best_scores_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trials, rewards, 'bo-', markersize=8, label='Mean Reward')
    ax.plot(trials, best, 'g--', linewidth=2, label='Best So Far')
    if best_so_far is not None:
        ax.axhline(y=best_so_far, color='r', linestyle=':', alpha=0.5, label=f'Final Best: {best_so_far:.4f}')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Bayesian Optimization: Best Observed Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = f"{output_dir}/{timestamp}-best-scores.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved best scores plot to {output_path}")
    plt.close()

    # Save history to CSV
    history_df = pd.DataFrame(best_scores_history)
    history_df.to_csv(f"{output_dir}/best_scores_history.csv", index=False)
    print(f"[INFO] Saved best scores history to {output_dir}/best_scores_history.csv")


def save_timing_summary(total_duration, max_trials):
    """Save timing summary to CSV."""
    if not best_scores_history:
        return
    
    trial_durations = [h.get('duration_seconds') for h in best_scores_history if 'duration_seconds' in h]
    
    timing_data = {
        'total_duration_seconds': total_duration,
        'total_duration': format_duration(total_duration),
        'num_trials_completed': len([d for d in trial_durations if d is not None]),
        'num_trials_requested': max_trials,
        'avg_trial_duration_seconds': np.mean(trial_durations) if trial_durations else None,
        'avg_trial_duration': format_duration(np.mean(trial_durations)) if trial_durations else None,
        'min_trial_duration_seconds': np.min(trial_durations) if trial_durations else None,
        'max_trial_duration_seconds': np.max(trial_durations) if trial_durations else None,
        'start_time': datetime.fromtimestamp(run_start_time).strftime('%Y-%m-%d %H:%M:%S') if run_start_time else None,
        'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    timing_df = pd.DataFrame([timing_data])
    timing_df.to_csv(f"{run_dir}/timing_summary.csv", index=False)
    print(f"[INFO] Saved timing summary to {run_dir}/timing_summary.csv")


def optimise(param_set=None, max_trials=1, learning_steps=None, cycles=None, seeds=None):
    """
    Bayesian optimization using Ax with qLogNoisyExpectedImprovement acquisition function.
    """
    global best_scores_history, best_so_far, run_start_time
    best_scores_history = []
    best_so_far = None
    run_start_time = time.time()

    if seeds is None:
        seeds = SEEDS
    if learning_steps is None:
        learning_steps = 80_000
    if cycles is None:
        cycles = 16

    width = os.get_terminal_size().columns
    print(f"[INFO] BO run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for pair, env in list(product(pairs, ENVS)):
        def get_client():
            from ax.api.client import Client
            client = Client()
            alg1, alg2 = pair.split("-")
            params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
            params = params + [update_ratio_param]
            client.configure_experiment(name="sarl_opt", parameters=params)
            client.configure_optimization(objective="mean_reward")
            return client
        client = get_client()

        def get_executor():
            executor = AutoExecutor(folder=f"{run_dir}/submitit", cluster=cluster)
            executor.update_parameters(timeout_min=180)
            executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)
            return executor
        executor = get_executor()

        def objective_function(params, trial_index=None):
            if param_set is not None:
                print("[DEBUG] Using fixed parameters!")
                discrete_lr, continuous_lr, update_ratio = param_set
            else:
                discrete_lr = params['discrete_learning_rate']
                continuous_lr = params['continuous_learning_rate']
                update_ratio = params['update_ratio']

            job_name = f"trial_{trial_index}-{pair.replace('-', '_')}-{env}-{seeds}"
            mean_reward, mean_reward_se = run_training(
                discrete_lr, continuous_lr, update_ratio, seeds, job_name,
                run_subdir=f"trial_{trial_index}",
                learning_steps=learning_steps, cycles=cycles
            )
            return {"mean_reward": (mean_reward, mean_reward_se)}

        def run_parallel_exps():
            jobs = []
            global submitted_jobs
            submitted_jobs = 0

            while submitted_jobs < max_trials or jobs:
                def run_trials():
                    global submitted_jobs
                    trial_index_to_param = client.get_next_trials(
                        min(PARALLEL_LIMIT - len(jobs), max_trials - submitted_jobs)
                    )
                    for trial_index, parameters in trial_index_to_param.items():
                        print(f"[INFO] Submitting parameters: {parameters}")
                        job = executor.submit(objective_function, parameters, trial_index)
                        submitted_jobs += 1
                        jobs.append((job, trial_index, parameters, time.time()))
                        time.sleep(1)

                def learn_from_any_previous_trials():
                    for job, trial_index, params, start_time in jobs[:]:
                        if job.done() or type(job) in [LocalJob, DebugJob]:
                            result = job.result()
                            mean_reward = result['mean_reward'][0]
                            duration = time.time() - start_time
                            print(f"\n[JOB RESULT]: {result}")
                            print(f"[TIMING] Trial {trial_index} completed in {format_duration(duration)}")
                            print("-" * width)
                            track_best_score(mean_reward, trial_index, params, duration)
                            history_df = pd.DataFrame(best_scores_history)
                            history_df.to_csv(f"{run_dir}/best_scores_history.csv", index=False)
                            _ = client.complete_trial(trial_index=trial_index, raw_data=result)
                            save_client(client, wip=True)
                            _ = jobs.remove((job, trial_index, params, start_time))

                run_trials()
                learn_from_any_previous_trials()

            best = client.get_best_parameterization()
            save_client(client)
            return {"best": best}

        outcome = run_parallel_exps()
        print(f"\n[RESULT] {outcome['best'][0]} results in {outcome['best'][1]} observed on trial {outcome['best'][2]}")

        total_duration = time.time() - run_start_time
        print(f"\n[TIMING] BO run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[TIMING] Total duration: {format_duration(total_duration)}")
        print("-" * width)

        save_timing_summary(total_duration, max_trials)
        print("-" * width)
