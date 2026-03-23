"""
Bayesian optimization functions using Ax.
"""
import os
import time
from itertools import product
from pathlib import Path
from submitit import AutoExecutor, LocalJob, DebugJob

from config import (
    pairs, ENVS, SEEDS,
    MAX_TRIALS, PARALLEL_LIMIT,
    update_ratio_param, get_params_by_alg,
    CPU_CORES_PER_TASK, run_dir, cluster
)
from training import run_training


def save_client(client, wip=False):
    from datetime import datetime
    yyyy_mm_dd_hhmm = datetime.now().strftime("%Y-%m-%d_%H-%M")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    if wip:
        df = client.summarize()
        df.to_csv(f"{run_dir}/wip-{yyyy_mm_dd_hhmm}-client.csv", index=False)
    else:
        client.summarize().to_csv(f"{run_dir}/{yyyy_mm_dd_hhmm}-client.csv", index=False)
        client.save_to_json_file(f"{run_dir}/{yyyy_mm_dd_hhmm}-client.json")
    return True


def optimise(param_set=None, max_trials=1):
    """
    Bayesian optimization using Ax with qLogNoisyExpectedImprovement acquisition function.
    """
    width = os.get_terminal_size().columns

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

            job_name = f"trial_{trial_index}-{pair.replace('-', '_')}-{env}-{SEEDS}"
            mean_reward, mean_reward_se = run_training(
                discrete_lr, continuous_lr, update_ratio, SEEDS, job_name,
                run_subdir=f"trial_{trial_index}"
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
                        jobs.append((job, trial_index))
                        time.sleep(1)

                def learn_from_any_previous_trials():
                    for job, trial_index in jobs[:]:
                        if job.done() or type(job) in [LocalJob, DebugJob]:
                            result = job.result()
                            print(f"\n[JOB RESULT]: {result}")
                            print("-" * width)
                            _ = client.complete_trial(trial_index=trial_index, raw_data=result)
                            save_client(client, wip=True)
                            _ = jobs.remove((job, trial_index))

                run_trials()
                learn_from_any_previous_trials()

            best = client.get_best_parameterization()
            save_client(client)
            return {"best": best}

        outcome = run_parallel_exps()
        print(f"\n[RESULT] {outcome['best'][0]} results in {outcome['best'][1]} observed on trial {outcome['best'][2]}")
        print("-" * width)
