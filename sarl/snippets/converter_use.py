# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import RecordEpisodeStatistics  # Replaces deprecated Gymnasium Monitor wrapper
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CallbackList

from sarl.environments.wrappers.converter import PamdpToMdp, HybridPolicy
from sarl.agents.callbacks.data_callback import DataCallback

def _make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)
    return env


def run_converter(discreteAlg="", continuousAlg="", env_name:str="", discrete_only=None, continuousOnly=None,
                  max_steps=None, learning_steps=None, cycles=None, seeds=[1],
                  use_tensorboard:bool=True, write_csv:bool=True, origin_log_dir=None):
    '''Collect data by training a specified HybridPolicy on a given converted environment.'''
    assert not (discrete_only and continuousOnly)
    for seed in seeds:
        parent_log_dir = f"{origin_log_dir}/seed_{seed}/"

        pamdp = _make_env(env_name=env_name, seed=seed)
        # pamdp_eval = _make_env(env_name=env_name, seed=seed+1)

        class IgnoreStepCount(ObservationWrapper):
            def __init__(self, env: gym.Env):
                super().__init__(env)
                self.observation_space = self.observation_space[0]
            def observation(self, observation):
                return observation[0]
        if env_name in ["Platform-v0", "Goal-v0"]:
            pamdp = IgnoreStepCount(pamdp)  # Necessary yet unsure why

        mdp = PamdpToMdp(pamdp)
        mdp = RecordEpisodeStatistics(mdp)
        mdp = Monitor(mdp, filename=f"{parent_log_dir}/mdp-monitor/")  # Seemingly ineffectual

        # Agent setup
        log_dir = ""
        log_dir_discrete = ""
        log_dir_continuous = ""
        sb3_logger_discrete:Logger = None
        sb3_logger_continuous:Logger = None
        sharedDataCallback = DataCallback()
        if use_tensorboard or write_csv:
            # if discrete_only:
            #     log_dir = f"{parent_log_dir}/test_output/discrete-{discreteAlg.lower()}/{env_name.lower()}/{str(learning_steps)}steps"
            #     sb3_logger_discrete = configure(log_dir_discrete, ["stdout", "csv", "tensorboard"])
            # elif discrete_only is False:
            #     log_dir = f"{parent_log_dir}/test_output/continuous-{continuousAlg.lower()}/{env_name.lower()}/{str(learning_steps)}steps"
            #     sb3_logger_continuous = configure(log_dir_continuous, ["stdout", "csv", "tensorboard"])
            # else:
            log_dir_discrete = f"{parent_log_dir}/{discreteAlg.lower()}-{continuousAlg.lower()}-{env_name.lower()}-discrete-{cycles}cycles-{str(learning_steps)}steps/"
            sb3_logger_discrete = configure(log_dir_discrete, ["stdout", "csv", "tensorboard"])
            log_dir_continuous = f"{parent_log_dir}/{discreteAlg.lower()}-{continuousAlg.lower()}-{env_name.lower()}-continuous-{cycles}cycles-{str(learning_steps)}steps/"
            sb3_logger_continuous = configure(log_dir_continuous, ["stdout", "csv", "tensorboard"])

        if discrete_only:
            def continuousPolicy(x): return mdp.action_parameter_space.sample()
            discreteActionMDP = mdp.getComponentMdp(action_space_is_discrete=True, internal_policy=continuousPolicy)
            discreteAgent = {
                "PPO": PPO("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir),
                "A2C": A2C("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir)
            }[discreteAlg]
            discreteAgent.set_logger(sb3_logger_discrete)
            agent = HybridPolicy(discreteAgent=discreteAgent, continuousPolicy=continuousPolicy)
        elif continuousOnly is False:
            def discretePolicy(x): return mdp.discrete_action_space.sample()
            continuousActionMDP = mdp.getComponentMdp(action_space_is_discrete=False, internal_policy=discretePolicy, combine_continuous_actions=True)
            continuousAgent = {
                "PPO": PPO("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir),
            }[continuousAlg]
            agent = HybridPolicy(discretePolicy=discretePolicy, continuousAgent=continuousAgent)
            continuousAgent.set_logger(sb3_logger_continuous)
        else:
            discreteActionMDP = mdp.getComponentMdp(action_space_is_discrete=True)#, internal_policy=continuousAgent.predict)
            continuousActionMDP = mdp.getComponentMdp(action_space_is_discrete=False, combine_continuous_actions=True)#, internal_policy=discreteAgent.predict)

            discreteAgent = {
                "A2C": A2C("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_discrete),
                "DQN": DQN("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_discrete),
                "PPO": PPO("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_discrete),
            }
            discreteAgent = discreteAgent[discreteAlg]
            discreteAgent.set_logger(sb3_logger_discrete)

            continuousAgent = {
                "A2C": A2C("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_continuous),
                "DDPG": DDPG("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_continuous),
                "PPO": PPO("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_continuous),
                "SAC": SAC("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_continuous),
                "TD3": TD3("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_continuous),
            }[continuousAlg]
            continuousAgent.set_logger(sb3_logger_continuous)

            discreteActionMDP.internal_policy = lambda obs: continuousAgent.predict(obs)[0]
            continuousActionMDP.internal_policy = lambda obs: discreteAgent.predict(obs)[0]

            agent = HybridPolicy(discreteAgent=discreteAgent, continuousAgent=continuousAgent, name=f"{discreteAlg}-{continuousAlg}")

        callbacks = CallbackList([
            DataCallback(),
        ])
        agent.learn(learning_steps, cycles=cycles, callback=callbacks, progress_bar=True)

        # Evaluate TODO:
        # obs, info = mdp.reset()
        # for i in range(max_steps*2):
        #     obs, reward, terminated, truncated, info = mdp.step(agent.predict(obs))

    return True


# Functions for train.py
# TODO: Reduce duplication - Only pass run_converter to train.py, and have them call it correctly
# TODO: Implement train_episodes
def ppo_ppo_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="PPO", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_ppo_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="PPO", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_ppo_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="PPO", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)

def ppo_a2c_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="A2C", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_a2c_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="A2C", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_a2c_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="A2C", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)

def ppo_ddpg_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="DDPG", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_ddpg_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="DDPG", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_ddpg_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="DDPG", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)

def ppo_sac_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="SAC", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_sac_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="SAC", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_sac_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="SAC", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)

def ppo_td3_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="TD3", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_td3_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="TD3", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_td3_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="TD3", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)


def ppo_ppo_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="PPO", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_ppo_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="PPO", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_ppo_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="PPO", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)

def ppo_a2c_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="A2C", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_a2c_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="A2C", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_a2c_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="A2C", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)

def ppo_ddpg_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="DDPG", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_ddpg_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="DDPG", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_ddpg_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="DDPG", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)

def ppo_sac_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="SAC", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_sac_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="SAC", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_sac_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="SAC", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)

def ppo_td3_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="TD3", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def a2c_td3_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="TD3", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
def dqn_td3_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="TD3", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, origin_log_dir=output_dir)
