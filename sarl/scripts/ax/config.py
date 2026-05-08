"""
Configuration constants for Ax-driven hyperparameter optimization.

Contains:
    - Run path generation (get_run_path)
    - Execution settings (LOCAL_DEBUG_MODE, cluster, cpus)
    - Training hyperparameters (LEARNING_STEPS, CYCLES, SEEDS)
    - Grid search parameters (GRID_PARAMS with lr combinations)
    - BO parameters (BOUNDS_LR, BOUNDS_UPDATE_RATIO)
"""

from datetime import datetime
from itertools import product
from pathlib import Path

from ax.api.configs import RangeParameterConfig

ROOT_STR = "."
LOCAL_DEBUG_MODE = True


def get_run_path(run_type: str, run_scale: str, run_state: str = "incomplete") -> str:
    return f"{ROOT_STR}/runs/{run_type}/{run_scale}/{run_state}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"


cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"
CPU_CORES_PER_TASK = 1
HYDRA_CONFIG_PATH = "../../config"
TRAIN_EPISODES = 40_000
ON_POLICY_PARAMS = {"n_steps": 100}

BOUNDS_LR = (1e-6, 1e-2)
BOUNDS_UPDATE_RATIO = (0.01, 0.99)

SINGLE_PARAM_MODE = True
FIXED_CONTINUOUS_LR = 1e-3
FIXED_UPDATE_RATIO = 0.5

LS_DEBUG = 100  # Learning steps
CYC_DEBUG = 1  # Cycles
NUM_SEEDS_DEBUG = 1  # Consider 2, not too high as BO handles noise well
MAX_TRIALS_TEST = 3

LS_PROPER = 30_000  # Change from 80_000
CYC_PROPER = 4  # Change from 16
NUM_SEEDS = 1  # Consider 2, not too high as BO handles noise well
MAX_TRIALS = 500  # Change from 1

PARALLEL_LIMIT = (
    1  # Change from 5  # TODO: Consider changing before running on cluster!
)
LEARNING_STEPS = LS_PROPER
CYCLES = CYC_PROPER
BASE_SEED = 1000
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]
ROTATE_SEEDS_PER_TRIALS = (
    True  # Rotate seeds across trials for stronger statistical claims
)

# Quick test settings for local development
LS_TEST = LS_DEBUG
CYC_TEST = CYC_DEBUG
NUM_SEEDS_TEST = NUM_SEEDS_DEBUG
PARALLEL_LIMIT_TEST = 1  # Allows more parallelism for quick local test runs

ENVS = ["platform"]
DISCRETE_ALGS = ["ppo"] #["dqn"]
CONTINUOUS_ALGS = ["ppo"] #["sac"]

LR_LOW = 1e-4
LR_MED = 3.16e-3
LR_HIGH = 1e-2
GRID_PARAMS = [
    {"discrete_lr": LR_LOW, "continuous_lr": LR_LOW, "update_ratio": 0.5},
    {"discrete_lr": LR_MED, "continuous_lr": LR_LOW, "update_ratio": 0.5},
    {"discrete_lr": LR_HIGH, "continuous_lr": LR_LOW, "update_ratio": 0.5},
    {"discrete_lr": LR_LOW, "continuous_lr": LR_MED, "update_ratio": 0.5},
    {"discrete_lr": LR_MED, "continuous_lr": LR_MED, "update_ratio": 0.5},
    {"discrete_lr": LR_HIGH, "continuous_lr": LR_MED, "update_ratio": 0.5},
    {"discrete_lr": LR_LOW, "continuous_lr": LR_HIGH, "update_ratio": 0.5},
    {"discrete_lr": LR_MED, "continuous_lr": LR_HIGH, "update_ratio": 0.5},
    {"discrete_lr": LR_HIGH, "continuous_lr": LR_HIGH, "update_ratio": 0.5},
]

update_ratio_param = RangeParameterConfig(
    name="update_ratio",
    bounds=BOUNDS_UPDATE_RATIO,
    parameter_type="float",
    scaling="linear",
)


def get_params_by_alg(label: str = ""):
    shared_params = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=BOUNDS_LR,
            parameter_type="float",
            scaling="log",
        )
    ]
    return {
        "a2c": shared_params + [],
        "dqn": shared_params + [],
        "ppo": shared_params + [],
        "ddpg": shared_params + [],
        "sac": shared_params + [],
        "td3": shared_params + [],
    }


pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(DISCRETE_ALGS, CONTINUOUS_ALGS)]
