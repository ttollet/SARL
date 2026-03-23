#!/usr/bin/env python3
"""
Ax-driven hyperparameter optimization for SARL agents.

Usage:
    python main.py              # Run Bayesian optimization

References:
    - https://ax.dev/docs/0.5.0/tutorials/submitit/
    - https://ax.dev/docs/0.5.0/bayesopt/
"""

import os
import yaml
from pathlib import Path
from datetime import datetime

from ax.api.client import Client

from config import (
    DISCRETE_ALGS, CONTINUOUS_ALGS, ENVS,
    SEEDS, LEARNING_STEPS, CYCLES,
    TRAIN_EPISODES, ON_POLICY_PARAMS,
    update_ratio_param, get_params_by_alg,
    USE_GRID, run_dir
)
from grid_search import run_grid_search
from bayes_opt import optimise


def create_config_file(run_dir):
    """Save run configuration to YAML."""
    config = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M"),
        'settings': {
            'algorithm': f"{DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}",
            'environment': ENVS[0],
            'seeds': SEEDS,
            'learning_steps': LEARNING_STEPS,
            'cycles': CYCLES,
            'train_episodes': TRAIN_EPISODES,
            'on_policy_n_steps': ON_POLICY_PARAMS['n_steps']
        }
    }
    config_path = Path(run_dir) / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    create_config_file(run_dir)

    if USE_GRID:
        pair = f"{DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}"
        client = Client()
        alg1, alg2 = pair.split("-")
        params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
        params = params + [update_ratio_param]
        client.configure_experiment(name="sarl_grid", parameters=params)
        client.configure_optimization(objective="mean_reward")

        results_df = run_grid_search(client)
        print("[INFO] Grid search complete!")
    else:
        optimise(max_trials=1)
