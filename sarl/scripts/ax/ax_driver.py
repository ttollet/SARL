# To be run on a the Lancaster University HEC SLURM login node.

# Imports
import subprocess
import json
import time
from itertools import product
from collections.abc import Sequence
from typing import Tuple

from ax.api.types import TParameterization
import numpy as np

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

# Constants
MAX_TRIALS = 10  # Big effect on duration
PARALLEL_LIMIT = 40
PARAMS_BY_ALG = {
    "ppo": [
        RangeParameterConfig(
            name="___",
            bounds=(0, 1),  # TODO: Set
            parameter_type="float",
            scaling="log"
        ),
    ],
}
ENVS = ["platform"]  # ["platform", "goal"]

# Experiment Loop
pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(PARAMS_BY_ALG.keys(), repeat=2)]

for pair, env in list(product(pairs, ENVS)):
    output = {}

    # STEP 1 - Create the experiment
    client = Client()
    alg1, alg2 = pair.split("-")
    params = PARAMS_BY_ALG[alg1] + PARAMS_BY_ALG[alg2]
    client.configure_experiment(name="sarl_opt", parameters=params)
    client.configure_optimization(objective="mean_reward")

    trial_index = 0
    # STEP 2 - Submit jobs to HEC
    while trial_count < MAX_TRIALS:
        trials = client.get_next_trials(max_trials=np.round(0.1 * MAX_TRIALS))
        # Regarding parallelism: https://ax.dev/docs/0.5.0/bayesopt/#tradeoff-between-parallelism-and-total-number-of-trials
        for trial_index, params in trials:
            # Write params to a JSON file accessible by script
            param_file = f"results/trial_{trial_index}.json"
            with open(param_file, "w") as f:
                json.dump(params, f)

            # Submit a SLURM job for this trial
            _ = subprocess.run(["sbatch", "evaluate_trial.sh", str(trial_index)])
            trial_count += 1

    # STEP 3 - Monitor until trials complete
    while len(client.get_trials_data_frame()) < 10:
        for i in range(10):
            result_path = f"results/result_{i}.json"
            try:
                with open(result_path) as f:
                    result = json.load(f)
                    client.complete_trial(
                        trial_index=i, raw_data=(result["reward"], 0.0)
                    )
            except FileNotFoundError:
                pass
        time.sleep(60 * 5)
    _ = client._generation_strategy  # TODO: Save this
    best_parameters, prediction, index, name = client.get_best_parameterization()  # TODO: Save this
    print("Best Parameters:", best_parameters)
    print("Prediction (mean, variance):", prediction)
