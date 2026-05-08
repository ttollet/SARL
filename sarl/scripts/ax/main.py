#!/usr/bin/env python3
"""
Ax-driven hyperparameter optimization for SARL agents.

Usage:
    python main.py                      # Run Bayesian optimization (proper)
    python main.py --grid               # Run grid search (proper)
    python main.py --debug                # Run Bayesian optimization (debug)
    python main.py --grid --debug         # Run grid search (debug)
    python main.py --grid --wandb         # Run grid search with wandb logging

References:
    - https://ax.dev/docs/0.5.0/tutorials/submitit/
    - https://ax.dev/docs/0.5.0/bayesopt/
"""

import argparse
import shutil
import yaml
from pathlib import Path
from datetime import datetime

from ax.api.client import Client

from config import (
    DISCRETE_ALGS,
    CONTINUOUS_ALGS,
    ENVS,
    MAX_TRIALS,
    SEEDS,
    LEARNING_STEPS,
    CYCLES,
    TRAIN_EPISODES,
    ON_POLICY_PARAMS,
    update_ratio_param,
    get_params_by_alg,
    get_run_path,
    LS_TEST,
    CYC_TEST,
    NUM_SEEDS_TEST,
    MAX_TRIALS_TEST,
    PARALLEL_LIMIT,
    PARALLEL_LIMIT_TEST,
)
from grid_search import run_grid_search
from bayes_opt import optimise, plot_best_scores


def create_config_file(run_dir, learning_steps, cycles, seeds):
    """Save run configuration to YAML."""
    config = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "settings": {
            "algorithm": f"{DISCRETE_ALGS[0]}-{CONTINUOUS_ALGS[0]}",
            "environment": ENVS[0],
            "seeds": seeds,
            "learning_steps": learning_steps,
            "cycles": cycles,
            "train_episodes": TRAIN_EPISODES,
            "on_policy_n_steps": ON_POLICY_PARAMS["n_steps"],
        },
    }
    config_path = Path(run_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def mark_run_complete(src_path: str) -> Path:
    """Move run from incomplete to parent directory."""
    src = Path(src_path)
    dst = src.parent.parent / src.name
    shutil.move(src, dst)
    print(f"[INFO] Moved run to {dst}")
    return dst


if __name__ == "__main__":
    assert Path.cwd().name == "ax"
    parser = argparse.ArgumentParser(
        description="Ax-driven hyperparameter optimization"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug/test settings (few seeds, short training)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Use grid search (default: Bayesian optimization)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    args = parser.parse_args()

    run_type = "grid" if args.grid else "bayesian"
    run_scale = "debug" if args.debug else "proper"
    run_dir = get_run_path(run_type, run_scale, "incomplete")

    # Determine settings
    if args.debug:
        learning_steps = LS_TEST
        cycles = CYC_TEST
        seeds = [1000 + i for i in range(NUM_SEEDS_TEST)]
        max_trials = MAX_TRIALS_TEST
        parallel_limit = PARALLEL_LIMIT_TEST
        print(
            f"[DEBUG MODE] learning_steps={learning_steps}, cycles={cycles}, seeds={seeds}, max_trials={max_trials}, parallel_limit={parallel_limit}"
        )
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
        params = (
            get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
        )
        params = params + [update_ratio_param]
        client.configure_experiment(name="sarl_grid", parameters=params)
        client.configure_optimization(objective="mean_reward")

        results_df = run_grid_search(
            client,
            learning_steps=learning_steps,
            cycles=cycles,
            seeds=seeds,
            wandb_enabled=args.wandb,
            output_dir=run_dir,
        )
        mark_run_complete(run_dir)
    else:
        optimise(
            max_trials=max_trials,
            learning_steps=learning_steps,
            cycles=cycles,
            seeds=seeds,
            parallel_limit=parallel_limit,
            output_dir=run_dir,
        )
        # Plot best scores after BO completes
        plot_best_scores(run_dir)
        mark_run_complete(run_dir)
