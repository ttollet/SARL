# To be run on a SLURM login node.
#
# Based on advice from:
# - https://ax.dev/docs/0.5.0/tutorials/submitit/
# - https://ax.dev/docs/0.5.0/bayesopt/#tradeoff-between-parallelism-and-total-number-of-trials

# %% Setup
# Imports
import time
import warnings
from itertools import product
from profile import run
from statistics import mean

import numpy as np
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from submitit import AutoExecutor, DebugJob, LocalJob
from tqdm import tqdm

from sarl.train import main

warnings.filterwarnings("ignore")  # TODO: Ensure works

# Constants
# - set every time:
LOCAL_DEBUG_MODE = True  # TODO: Set to True for local debugging
SUBMITIT_DIR = "submitit"
HYDRA_CONFIG_PATH = "../../config"
# - computation:
CPU_CORES_PER_TASK = 4
MAX_TRIALS = 40  # Big effect on duration
PARALLEL_LIMIT = 2  # 40
TRAIN_EPISODES = 100  # 1000000
# - common bounds
MIN_LR = 0  # 1e-6
MAX_LR = 1  # 1e-3
MIN_GAMMA = 0.9
MAX_GAMMA = 0.9999
# - misc:
SEEDS = [42]
ENVS = ["platform"]  # TODO: Set correct final ENVS
# ["platform", "goal"]
UPDATE_RATIO_PARAM = RangeParameterConfig(
    name="update_ratio",
    bounds=(0.1, 0.4),  # TODO: Go back to original
    # bounds=(0.1, 0.9),  # Original
    parameter_type="float",
    scaling="linear",
)
# TODO: Set correct final *_ALGS
DISCRETE_ALGS = ["ppo"]
CONTINUOUS_ALGS = ["ppo"]  # TODO: Vary discrete alg choice first
# DISCRETE_ALGS = ["a2c", "dqn", "ppo"]
# CONTINUOUS_ALGS = ["a2c", "ddpg", "ppo", "sac", "td3"]

# %% ---- Toy Approach ----
# INFO: Increased training episodes results in decreased optimal learning rate
# DONE: Training is not seperated from evaluation
# WARN: Training time is insufficient to reach theorectical convergence (~0.9 mean_reward)
"""
Requires stable_baselines3, gymnasium, ax-platform, matplotlib.
"""
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from sarl.common.bester.agents.pdqn import PDQNAgent
from sarl.common.bester.common.platform_domain import PlatformFlattenedActionWrapper
from sarl.common.bester.common.wrappers import (
    ScaledParameterisedActionWrapper,
    ScaledStateWrapper,
)
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv

# config
max_steps = 500
train_episodes = 2000  # 20_000 # 2_500 # int(10_000 / 4)
test_episodes = 1_000
optim_steps = 5  # 20
max_trials = 4  # 5


def _get_mean_reward(
    train_episodes, test_episodes, agent, env, max_steps, seed, reward_scale=1
):
    """Train to output a list of returns by timestep."""

    def pad_action(act, act_param):
        params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
        params[act] = act_param
        return (act, params)

    returns = []
    for episodes in [train_episodes, test_episodes]:
        returns = []
        for _ in tqdm(
            range(episodes),
            desc=f"{'Training' if episodes == train_episodes else 'Testing'}",
        ):
            seed += 1
            (observation, steps), _ = env.reset(seed=seed)
            observation = np.array(observation, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(observation)
            action = pad_action(act, act_param)

            # Episode loop
            agent.start_episode()
            episode_return = 0
            last_timestep = 0
            for timestep in range(max_steps):
                last_timestep += 1
                (next_observation, steps), reward, terminated, truncated, info = (
                    env.step(action)
                )
                next_observation = np.array(
                    next_observation, dtype=np.float32, copy=False
                )
                next_act, next_act_param, next_all_action_parameters = agent.act(
                    next_observation
                )
                next_action = pad_action(next_act, next_act_param)

                episode_return += reward
                r = reward * reward_scale

                agent.step(
                    observation,
                    (act, all_action_parameters),
                    r,
                    next_observation,
                    (next_act, next_all_action_parameters),
                    terminated or truncated,
                    steps,
                )
                act, act_param, all_action_parameters = (
                    next_act,
                    next_act_param,
                    next_all_action_parameters,
                )
                action = next_action
                observation = next_observation
                if terminated or truncated:
                    break
            agent.end_episode()
            returns.append(episode_return)
    env.close()
    return mean(returns)


def _init_agent(env, seed, lr=0.001):
    pdqn_setup = {
        "observation_space": env.observation_space.spaces[0],
        "action_space": env.action_space,
        # "learning_rate_actor": 0.001,  # 0.0001
        "learning_rate_actor": lr,
        "learning_rate_actor_param": 0.00001,  # 0.001
        "epsilon_steps": 1000,
        "epsilon_final": 0.01,
        "gamma": 0.95,
        "clip_grad": 1,
        "indexed": False,
        "average": False,
        "random_weighted": False,
        "tau_actor": 0.1,
        "weighted": False,
        "tau_actor_param": 0.001,
        "initial_memory_threshold": 128,
        "use_ornstein_noise": True,
        "replay_memory_size": 20000,
        "inverting_gradients": True,
        "actor_kwargs": {"hidden_layers": (128,), "action_input_layer": 0},
        "actor_param_kwargs": {
            "hidden_layers": (128,),
            "output_layer_init_std": 1e-4,
            "squashing_function": False,
        },
        "zero_index_gradients": False,
        "seed": seed,
    }
    return PDQNAgent(**pdqn_setup)


def _make_env(env_name: str, max_steps: int, seed: int):
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env, deque_size=max_steps)  # Note stats
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)

    # Setup from pdqn_use.py
    initial_params_ = [3.0, 10.0, 400.0]
    for a in range(env.action_space.spaces[0].n):
        initial_params_[a] = (
            2.0
            * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low)
            / (
                env.action_space.spaces[1].spaces[a].high
                - env.action_space.spaces[1].spaces[a].low
            )
            - 1.0
        )
    # initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
    initial_bias = np.zeros(env.action_space.spaces[0].n)
    for a in range(env.action_space.spaces[0].n):
        initial_bias[a] = initial_params_[a]
    ## Env Wrappers | TODO: Convert to environment options
    env = PlatformFlattenedActionWrapper(env)
    env = ScaledParameterisedActionWrapper(env)  # Parameters -> [-1,1]
    env = ScaledStateWrapper(env)  # Observations -> [-1,1

    return env


# init
# TODO: Specify max_steps, seed
client = Client()
parameters = [
    RangeParameterConfig(
        # name="learning_rate", parameter_type="float", bounds=(0, 0.1), scaling="linear"
        name="learning_rate",
        parameter_type="float",
        bounds=(0, 1),
        scaling="linear",
    )
]
client.configure_experiment(
    name="cartpole-ppo", parameters=parameters
)  # INFO: What to guess [A]
client.configure_optimization(objective="mean_reward")  # INFO: What to optimise [B]

# loop
mean_rewards = []
learning_rates = []
for _ in tqdm(range(optim_steps), desc="Optimizing"):
    trials = client.get_next_trials(max_trials=max_trials)  # INFO: Ax makes guesses [C]
    for trial_index, parameters in trials.items():
        # trial_index = 0
        # parameters = {"learning_rate": 0.01}
        env = _make_env(env_name="Platform-v0", max_steps=max_steps, seed=trial_index)
        agent = _init_agent(env, trial_index, lr=parameters["learning_rate"])
        mean_reward = _get_mean_reward(
            train_episodes=train_episodes,
            test_episodes=test_episodes,
            agent=agent,
            env=env,
            max_steps=max_steps,
            seed=trial_index,
        )
        mean_rewards.append(mean_reward)
        learning_rates.append(parameters["learning_rate"])
        _trial_status = (
            client.complete_trial(  # INFO: Ax learns how the guesses went [D]
                trial_index=trial_index, raw_data={"mean_reward": mean_reward}
            )
        )

# %% output
best = client.get_best_parameterization()
print(best)  # 0.006 -> 0.22

# %% Debug
sorted_lr  # Store trial observed on
# %% Outcome
# plt.plot(mean_rewards)
# plt.xlabel("Iteration")
sorted_values = sorted(zip(learning_rates, mean_rewards))
sorted_lr, sorted_mean_rewards = zip(*sorted_values)
plt.plot(sorted_lr, sorted_mean_rewards)
plt.xlabel("Learning Rate")
plt.ylabel("Mean Reward")
plt.savefig("mean_rewards_rl.png")
# %%
plt.show()  # TODO: x=iteration, y=selected_point(mean reward)
# TODO: show with crosses the points plotted
# TODO: color initial crosses differently
# TODO: set sensible small initial guesses for lr, set limit to 0.2
# TODO: Only focus on one pair for now
