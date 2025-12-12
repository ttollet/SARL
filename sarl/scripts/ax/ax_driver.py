# To be run on a SLURM login node.
#
# Based on advice from:
# - https://ax.dev/docs/0.5.0/tutorials/submitit/
# - https://ax.dev/docs/0.5.0/bayesopt/#tradeoff-between-parallelism-and-total-number-of-trials

# %% Setup
# Imports
from profile import run
import time
from itertools import product
import warnings
from statistics import mean

import numpy as np
from tqdm import tqdm
from submitit import AutoExecutor, LocalJob, DebugJob
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

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
PARALLEL_LIMIT = 2 # 40
TRAIN_EPISODES = int(1000000 / 10)
# - common bounds
MIN_LR = 0 #1e-6
MAX_LR = 1 #1e-3
MIN_GAMMA = 0.9
MAX_GAMMA = 0.9999
# - misc:
SEED = 42
ENVS = ["platform"]  # TODO: Set correct final ENVS
# ["platform", "goal"]
UPDATE_RATIO_PARAM = RangeParameterConfig(
    name="update_ratio",
    bounds=(0.1, 0.9),
    parameter_type="float",
    scaling="linear"
)
# TODO: Set correct final *_ALGS
DISCRETE_ALGS = ["ppo"]
CONTINUOUS_ALGS = ["ppo"]  # TODO: Vary discrete alg choice first
# DISCRETE_ALGS = ["a2c", "dqn", "ppo"]
# CONTINUOUS_ALGS = ["a2c", "ddpg", "ppo", "sac", "td3"]

# %% ---- Toy Approach ----
# INFO: Increased training episodes results in decreased optimal learning rate
# WARN: Training is not seperated from evaluation
# WARN: Training time is insufficient to reach theorectical convergence (~0.9 mean_reward)
"""
Requires stable_baselines3, gymnasium, ax-platform, matplotlib.
"""
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.agents.pdqn import PDQNAgent
from sarl.common.bester.common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from sarl.common.bester.common.platform_domain import PlatformFlattenedActionWrapper


# config
max_steps = 500
train_episodes = 2_500 # int(10_000 / 4)
test_episodes = 1_000
optim_steps = 5  #20


def _get_mean_reward(train_episodes, test_episodes, agent, env, max_steps, seed, reward_scale=1):
    '''Train to output a list of returns by timestep.'''
    def pad_action(act, act_param):
        params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
        params[act] = act_param
        return (act, params)
    returns = []
    for episodes in [train_episodes, test_episodes]:
        returns = []
        for _ in tqdm(range(episodes)):
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
                (next_observation, steps), reward, terminated, truncated, info = env.step(action)
                next_observation = np.array(next_observation, dtype=np.float32, copy=False)
                next_act, next_act_param, next_all_action_parameters = agent.act(next_observation)
                next_action = pad_action(next_act, next_act_param)

                episode_return += reward
                r = reward * reward_scale

                agent.step(observation, (act, all_action_parameters), r, next_observation, (next_act, next_all_action_parameters), terminated or truncated, steps)
                act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
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
        "actor_kwargs": {'hidden_layers': (128,),
                         'action_input_layer': 0},
        "actor_param_kwargs": {
            'hidden_layers': (128,),
            'output_layer_init_std': 1e-4,
            'squashing_function': False},
        "zero_index_gradients": False,
        "seed": seed
    }
    return PDQNAgent(**pdqn_setup)

def _make_env(env_name: str, max_steps: int, seed: int):
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env, deque_size=max_steps)  # Note stats
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)

    # Setup from pdqn_use.py
    initial_params_ = [3., 10., 400.]
    for a in range(env.action_space.spaces[0].n):
        initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                    env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
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
        name="learning_rate", parameter_type="float", bounds=(0, 1), scaling="linear"
    )]
client.configure_experiment(name="cartpole-ppo", parameters=parameters)  # INFO: What to guess [A]
client.configure_optimization(objective="mean_reward")  # INFO: What to optimise [B]

# loop
mean_rewards = []
learning_rates = []
for _ in range(optim_steps):
    trials = client.get_next_trials(max_trials=5)  # INFO: Ax makes guesses [C]
    for trial_index, parameters in trials.items():
        # trial_index = 0
        # parameters = {"learning_rate": 0.01}
        env = _make_env(env_name="Platform-v0", max_steps=max_steps, seed=trial_index)
        agent = _init_agent(env, trial_index, lr=parameters["learning_rate"])
        mean_reward = _get_mean_reward(
            train_episodes=train_episodes, test_episodes=test_episodes,
            agent=agent, env=env, max_steps=max_steps, seed=trial_index)
        mean_rewards.append(mean_reward)
        learning_rates.append(parameters["learning_rate"])
        _trial_status = client.complete_trial(  # INFO: Ax learns how the guesses went [D]
            trial_index=trial_index, raw_data={"mean_reward": mean_reward}
        )

# %% output
best = client.get_best_parameterization()
print(best)  # 0.006 -> 0.22

# %% Outcome
# plt.plot(mean_rewards)
# plt.xlabel("Iteration")
sorted_values = sorted(zip(learning_rates, mean_rewards))
sorted_lr, sorted_mean_rewards = zip(*sorted_values)
plt.plot(sorted_lr, sorted_mean_rewards)
plt.xlabel("Learning Rate")
plt.ylabel("Mean Reward")
plt.savefig("mean_rewards_rl.png")
plt.show()

# %% ---- SCRIPT START ----
quit()  # TODO: REMOVE THIS LINE
cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"

def get_params_by_alg(label:str = ""):
    shared_params = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=(MIN_LR, MAX_LR),
            parameter_type="float",
            scaling="linear"
        ),
    ]
    return {
        "a2c": shared_params + [],
        "dqn": shared_params + [],
        "ppo": shared_params + [],  # INFO: Consider batch size
        "ddpg": shared_params + [],
        "sac": shared_params + [],
        "td3": shared_params + [],
    }
# pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(get_params_by_alg().keys(), repeat=2)]
pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(DISCRETE_ALGS, CONTINUOUS_ALGS)]


# %% Optimisation
# INFO: From source code of ax: the aqcuisition function used is qLogNoisyExpectedImprovement in BoTorch.
# See line 29 of https://github.com/facebook/Ax/blob/main/ax/generators/torch/botorch_modular/utils.py
def optimise():
    for pair, env in list(product(pairs, ENVS)):
        def get_client():
            # Setup Experiment via Ax
            client = Client()
            alg1, alg2 = pair.split("-")
            params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
            params = params + [UPDATE_RATIO_PARAM]
            client.configure_experiment(name="sarl_opt", parameters=params)  # INFO: What to guess [A]
            client.configure_optimization(objective="mean_reward")  # INFO: What to optimise [B]
            return client
        client = get_client()

        def get_executor():
            # Setup SubmitIt
            executor = AutoExecutor(folder=SUBMITIT_DIR, cluster=cluster)
            executor.update_parameters(timeout_min=60) # Timeout of the slurm job. Not including slurm scheduling delay.
            executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)
            return executor
        executor = get_executor()

        # def objective_function(params: dict[str, list[RangeParameterConfig]]):
        def objective_function(params: dict[str, float]):
            # Define the Function to Optimise.
            # Calls main() from train.py with an automated hydra config.
            GlobalHydra.instance().clear()  # critical reset
            with initialize(config_path=HYDRA_CONFIG_PATH, job_name=(f"{pair.replace('-', '_')}-{env}-{SEED}")):
                cfg = compose(config_name="sarl", return_hydra_config=True, overrides=[
                        f"algorithm={pair}",
                        f"environment={env}",
                        f"parameters.seeds={SEED}",
                        f"parameters.train_episodes={TRAIN_EPISODES}",
                        f"parameters.alg_params.discrete_learning_rate={params['discrete_learning_rate']}",
                        f"parameters.alg_params.continuous_learning_rate={params['continuous_learning_rate']}",
                    ])
                HydraConfig.instance().set_config(cfg)  # manually register config
                mean_reward = main(cfg)  # TODO: [1] main() returns mean_reward
            return {"mean_reward": mean_reward}#, "std_reward": std_reward}

        # This demonstrates the objective_function works
        # def toy_func():
        #     params = {"learning_rate": 0.5}
        #     print(objective_function(params))
        #     return True
        # toy_func()

        def run_parallel_exps():
            # Run the Experiment
            """
            Returns list of:
                - The parameters predicted to have the best optimization value without
                    violating any outcome constraints.
                - The metric values for the best parameterization. Uses model prediction if
                    ``use_model_predictions=True``, otherwise returns observed data.
                - The trial which most recently ran the best parameterization
                - The name of the best arm (each trial has a unique name associated with
                    each parameterization)
            """
            jobs = []
            global submitted_jobs
            submitted_jobs = 0
            while submitted_jobs < MAX_TRIALS or jobs:
                def run_trials():
                    global submitted_jobs
                    trial_index_to_param = client.get_next_trials(  # INFO: Ax makes guesses [C]
                        min(PARALLEL_LIMIT - len(jobs), MAX_TRIALS - submitted_jobs)
                    )
                    for trial_index, parameters in trial_index_to_param.items():
                        job = executor.submit(objective_function, parameters)
                        submitted_jobs += 1
                        jobs.append((job, trial_index))
                        time.sleep(1)
                def learn_from_any_previous_trials():
                    for job, trial_index in jobs[:]:  # INFO: Ax learns how any previous guesses went [D]
                        # Monitor for completed jobs
                        if job.done() or type(job) in [LocalJob, DebugJob]:
                            results = job.result()

                            print(results)  # TODO: Check working well
                            _ = client.complete_trial(trial_index=trial_index, raw_data=results)
                            _ = jobs.remove((job, trial_index))
                        # WARN: Reintroduce sleep() for Slurm
                        # time.sleep(1)
                run_trials()
                learn_from_any_previous_trials()
                # WARN: Reintroduce sleep() for Slurm
                # time.sleep(30)
            best = client.get_best_parameterization()  # TODO: Save best parameterisations & corresponding mean rewards
            # best_param, best_mean_reward = 0, 0
            return best
        outcome = run_parallel_exps()
        print(f"\n[RESULT] {outcome[0]} results in {outcome[1]} observed on trial {outcome[2]}")

        def visualise():  # TODO: Save visualisations
            pass
optimise()
