"""
Configuration constants for Ax-driven hyperparameter optimization.
"""
from datetime import datetime
from itertools import product
from pathlib import Path

from ax.api.configs import RangeParameterConfig

ROOT_STR = "."
LOCAL_DEBUG_MODE = True

yyyy_mm_dd_hhmm = datetime.now().strftime("%Y-%m-%d_%H-%M")
run_dir = f"{ROOT_STR}/runs/{yyyy_mm_dd_hhmm}"
Path(run_dir).mkdir(parents=True, exist_ok=True)

cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"
CPU_CORES_PER_TASK = 1
HYDRA_CONFIG_PATH = "../../config"
TRAIN_EPISODES = 40_000
ON_POLICY_PARAMS = {"n_steps": 100}

BOUNDS_LR = (1e-6, 1e-2)
BOUNDS_UPDATE_RATIO = (0.01, 0.99)

LS_TOY = 1000
CYC_TOY = 2
LS_MIN = 10_000
CYC_MIN = 8
LS_PROPER = 80_000
CYC_PROPER = 16

MAX_TRIALS = 1
PARALLEL_LIMIT = 150
LEARNING_STEPS = LS_PROPER
CYCLES = CYC_PROPER
NUM_SEEDS = 15
BASE_SEED = 1000
SEEDS = [BASE_SEED + i for i in range(NUM_SEEDS)]

ENVS = ["platform"]
DISCRETE_ALGS = ["dqn"]
CONTINUOUS_ALGS = ["sac"]

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

USE_GRID = True

update_ratio_param = RangeParameterConfig(
    name="update_ratio",
    bounds=BOUNDS_UPDATE_RATIO,
    parameter_type="float",
    scaling="linear"
)


def get_params_by_alg(label: str = ""):
    shared_params = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=BOUNDS_LR,
            parameter_type="float",
            scaling="log"
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
