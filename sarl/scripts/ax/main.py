#!/usr/bin/env python3
"""
Ax-driven hyperparameter optimization for SARL agents.

Usage:
    python main.py                      # Run Bayesian optimization
    python main.py --grid               # Run grid search
    python main.py --test               # Quick local test (few seeds, short training)
    python main.py --grid --test        # Quick local grid search test

References:
    - https://ax.dev/docs/0.5.0/tutorials/submitit/
    - https://ax.dev/docs/0.5.0/bayesopt/
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime

from ax.api.client import Client

from config import (
    DISCRETE_ALGS, CONTINUOUS_ALGS, ENVS, MAX_TRIALS,
    SEEDS, LEARNING_STEPS, CYCLES,
    TRAIN_EPISODES, ON_POLICY_PARAMS,
    update_ratio_param, get_params_by_alg,
    run_dir,
    LS_TEST, CYC_TEST, NUM_SEEDS_TEST, MAX_TRIALS_TEST,
    PARALLEL_LIMIT, PARALLEL_LIMIT_TEST
)
from grid_search import run_grid_search
from bayes_opt import optimise, plot_best_scores


def create_config_file(run_dir, learning_steps, cycles, seeds):
    """Save run configuration to YAML."""
    config = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M"),
        'settings': {
            'algorithm': f"{DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}",
            'environment': ENVS[0],
            'seeds': seeds,
            'learning_steps': learning_steps,
            'cycles': cycles,
            'train_episodes': TRAIN_EPISODES,
            'on_policy_n_steps': ON_POLICY_PARAMS['n_steps']
        }
    }
    config_path = Path(run_dir) / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ax-driven hyperparameter optimization")
    parser.add_argument("--grid", action="store_true", help="Run grid search instead of Bayesian optimization")
    parser.add_argument("--test", action="store_true", help="Use quick test settings (few seeds, short training)")
    args = parser.parse_args()

    # Determine settings
    if args.test:
        learning_steps = LS_TEST
        cycles = CYC_TEST
        seeds = [1000 + i for i in range(NUM_SEEDS_TEST)]
        max_trials = MAX_TRIALS_TEST
        parallel_limit = PARALLEL_LIMIT_TEST
        print(f"[TEST MODE] learning_steps={learning_steps}, cycles={cycles}, seeds={seeds}, max_trials={max_trials}, parallel_limit={parallel_limit}")
    else:
        learning_steps = LEARNING_STEPS
        cycles = CYCLES
        seeds = SEEDS
        max_trials = MAX_TRIALS
        parallel_limit = PARALLEL_LIMIT

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    create_config_file(run_dir, learning_steps, cycles, seeds)

    if args.grid:
        pair = f"{DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}"
        client = Client()
        alg1, alg2 = pair.split("-")
        params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
        params = params + [update_ratio_param]
        client.configure_experiment(name="sarl_grid", parameters=params)
        client.configure_optimization(objective="mean_reward")

        results_df = run_grid_search(client, learning_steps=learning_steps, cycles=cycles, seeds=seeds)
        print("[INFO] Grid search complete!")
    else:
        optimise(max_trials=max_trials, learning_steps=learning_steps, cycles=cycles, seeds=seeds, parallel_limit=parallel_limit)
        # Plot best scores after BO completes
        plot_best_scores(run_dir)
