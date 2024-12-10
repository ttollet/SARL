# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics  # Note stats
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor

from sarl.environments.wrappers.converter import PamdpToMdp, HybridPolicy
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.environments.gym_goal.envs import GoalEnv


def _make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)
    return env


def run_converter(discreteAlg:str=None, continuousAlg=None, env_name:str=None, discrete_only=None, continuousOnly=None,
                  max_steps:int=None, learning_steps:int=None, cycles=None, seeds=[1],
                  use_tensorboard:bool=False, write_csv:bool=False, parent_log_dir:str=None):
    '''Collect data by training a specified HybridPolicy on a given converted environment.'''
    assert not (discrete_only and continuousOnly)
    for seed in seeds:
        pamdp = _make_env(env_name=env_name, seed=seed)
        if env_name in ["Platform-v0", "Goal-v0"]:
            class IgnoreStepCount(ObservationWrapper):
                def __init__(self, env: gym.Env):
                    super().__init__(env)
                    self.observation_space = self.observation_space[0]
                def observation(self, observation):
                    return observation[0]
            pamdp = IgnoreStepCount(pamdp)
        mdp = PamdpToMdp(pamdp)

        # Agent setup
        log_dir = None
        log_dir_discrete = None
        log_dir_continuous = None
        if use_tensorboard or write_csv:
            if discrete_only:
                log_dir = f"{parent_log_dir}/test_output/discrete/{discreteAlg.lower()}/{env_name.lower()}/{str(learning_steps)}steps"
            elif discrete_only is False:
                log_dir = f"{parent_log_dir}/test_output/continuous/{continuousAlg.lower()}/{env_name.lower()}/{str(learning_steps)}steps"
            else:
                log_dir_discrete = f"{parent_log_dir}/test_output/hybrid/{discreteAlg.lower()}-{continuousAlg.lower()}-{cycles}-discrete/{env_name.lower()}/{str(learning_steps)}steps" 
                log_dir_continuous = f"{parent_log_dir}/test_output/hybrid/{discreteAlg.lower()}-{continuousAlg.lower()}-{cycles}-continuous/{env_name.lower()}/{str(learning_steps)}steps" 

        if discrete_only:
            def continuousPolicy(x): return mdp.action_parameter_space.sample()
            discreteActionMDP = mdp.getComponentMdp(action_space_is_discrete=True, internal_policy=continuousPolicy)
            discreteAgent = {
                "PPO": PPO("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir),
                "A2C": A2C("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir)
            }[discreteAlg]
            agent = HybridPolicy(discreteAgent=discreteAgent, continuousPolicy=continuousPolicy)
        elif continuousOnly is False:
            def discretePolicy(x): return mdp.discrete_action_space.sample()
            continuousActionMDP = mdp.getComponentMdp(action_space_is_discrete=False, internal_policy=discretePolicy, combine_continuous_actions=True)
            continuousAgent = {
                "PPO": PPO("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir),
            }[continuousAlg]
            agent = HybridPolicy(discretePolicy=discretePolicy, continuousAgent=continuousAgent)
        else:
            discreteActionMDP = mdp.getComponentMdp(action_space_is_discrete=True)#, internal_policy=continuousAgent.predict)
            continuousActionMDP = mdp.getComponentMdp(action_space_is_discrete=False, combine_continuous_actions=True)#, internal_policy=discreteAgent.predict)
            discreteAgent = {
                "PPO": PPO("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_discrete),
                "A2C": A2C("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_discrete),
            }
            discreteAgent = discreteAgent[discreteAlg]
            continuousAgent = {
                "PPO": PPO("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_continuous),
            }[continuousAlg]
            discreteActionMDP.internal_policy = lambda obs: continuousAgent.predict(obs)[0]
            continuousActionMDP.internal_policy = lambda obs: discreteAgent.predict(obs)[0]

            agent = HybridPolicy(discreteAgent=discreteAgent, continuousAgent=continuousAgent)

        agent.learn(learning_steps, cycles=cycles, progress_bar=True)

        # A few steps of the trained model
        obs, info = mdp.reset()
        for i in range(max_steps*2):
            # Maybe phrase assertion in terms of type? Uncertain
            # assert obs in mdp.observation_space # TODO: Achieve `obs in mdp.observation_space` for Goal-v0
            if mdp.expectingDiscreteAction():
                obs, reward, terminated, truncated, info = mdp.step(agent.predict(obs))
            else:
                obs, reward, terminated, truncated, info = mdp.step(agent.predict(obs))
    return True


# Functions for train.py
# TODO: Reduce duplication
# TODO: Implement train_episodes
env_name = "Platform-v0"
def ppo_ppo_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="PPO", env_name=env_name, max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, parent_log_dir=output_dir)
def a2c_ppo_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="PPO", env_name=env_name, max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, parent_log_dir=output_dir)

env_name = "Goal-v0"
def ppo_ppo_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="PPO", env_name=env_name, max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, parent_log_dir=output_dir)
def a2c_ppo_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="PPO", env_name=env_name, max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=True, write_csv=True, parent_log_dir=output_dir)
