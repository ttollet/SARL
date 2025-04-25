# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics  # Replaces deprecated Gymnasium Monitor wrapper
from stable_baselines3 import PPO, A2C, DQN, DDPG, SAC, TD3
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import CallbackList

from sarl.environments.wrappers.converter import PamdpToMdp, HybridPolicy
from sarl.agents.callbacks.data_callback import DataCallback


def _getMDP(env_name, pamdp_seed):
    '''Initialise and return a derived MDP from the chosen PAMDP'''
    env_pamdp = gym.make(env_name)
    env_pamdp.seed(pamdp_seed)  # Remove stochasticity
    np.random.seed(pamdp_seed)
    class IgnoreStepCount(ObservationWrapper):
        def __init__(self, env: gym.Env):
            super().__init__(env_pamdp)
            self.observation_space = self.observation_space[0]
        def observation(self, observation):
            return observation[0]
    if env_name in ["Platform-v0", "Goal-v0"]:
        env_pamdp = IgnoreStepCount(env_pamdp)  # Necessary yet unsure why
    mdp_env = PamdpToMdp(env_pamdp)
    mdp_env = RecordEpisodeStatistics(mdp_env)
    return mdp_env


def _init_seed_logging(origin_log_dir, seed, env_name, write_stdout, write_csv,
    use_tensorboard, discreteAlg, continuousAlg, cycles, learning_steps):
    '''Return required logging variables'''
    parent_log_dir = f"{origin_log_dir}/seed_{seed}/"
    log_dir = ""
    log_dir_discrete = ""
    log_dir_continuous = ""
    sb3_logger_discrete:Logger = None
    sb3_logger_continuous:Logger = None
    OUTPUT_OPTIONS = ["stdout", "csv", "tensorboard"]
    output_destinations = []
    for i in range(3):
        if [write_stdout, write_csv, use_tensorboard][i]:
            output_destinations.append(OUTPUT_OPTIONS[i])
    if use_tensorboard or write_csv:
        log_dir_discrete = f"{parent_log_dir}/{discreteAlg.lower()}-{continuousAlg.lower()}-{env_name.lower()}-discrete-{cycles}cycles-{str(learning_steps)}steps/"
        sb3_logger_discrete = configure(log_dir_discrete, output_destinations)
        log_dir_continuous = f"{parent_log_dir}/{discreteAlg.lower()}-{continuousAlg.lower()}-{env_name.lower()}-continuous-{cycles}cycles-{str(learning_steps)}steps/"
        sb3_logger_continuous = configure(log_dir_continuous, output_destinations)
        log_dir = f"{parent_log_dir}/{discreteAlg.lower()}-{continuousAlg.lower()}-{env_name.lower()}-evaluation-{cycles}cycles-{str(learning_steps)}steps/"
    return (sb3_logger_discrete, sb3_logger_continuous, log_dir,
        log_dir_discrete, log_dir_continuous)


def _getAgent(discrete_only, continuousOnly, mdp, seed, discreteAlg, logging_info,
    continuousAlg, env_name):
    '''Return agent to train'''
    (sb3_logger_discrete, sb3_logger_continuous, log_dir, log_dir_discrete,
        log_dir_continuous) = logging_info
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
        agent = HybridPolicy(discreteAgent=discreteAgent, continuousAgent=continuousAgent,
            name=f"{discreteAlg}-{continuousAlg}", env_name=env_name, seed=seed)
    return agent


def run_converter(discreteAlg="", continuousAlg="", env_name="", discrete_only=None,
    continuousOnly=None, max_steps=None, learning_steps=None, cycles=None, seeds=[1],
    use_tensorboard=False, write_csv=True, write_stdout=False, origin_log_dir=None,
    evaluation_interval=1):
    '''Collect data by training a specified HybridPolicy on a given environment
    via conversion.'''
    assert not (discrete_only and continuousOnly)
    for seed in seeds:
        mdp = _getMDP(env_name, seed)
        eval_mdp = _getMDP(env_name, seed+1)
        logging_info = _init_seed_logging(origin_log_dir, seed, env_name,
                write_stdout, write_csv, use_tensorboard, discreteAlg, continuousAlg,
                cycles, learning_steps)
        agent = _getAgent(discrete_only, continuousOnly, mdp, seed, discreteAlg,
            logging_info, continuousAlg, env_name)
        callbacks = CallbackList([DataCallback()])
        agent.learn(learning_steps, cycles=cycles, callback=callbacks, progress_bar=True,
            evaluation_interval=evaluation_interval, eval_mdp=eval_mdp, log_dir=origin_log_dir)
    return True


# Functions for train.py
# TODO: Reduce duplication - Only pass run_converter to train.py, and have them call it correctly
# TODO: Implement train_episodes
def ppo_ppo_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="PPO", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_ppo_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="PPO", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_ppo_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="PPO", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)

def ppo_a2c_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="A2C", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_a2c_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="A2C", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_a2c_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="A2C", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)

def ppo_ddpg_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="DDPG", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_ddpg_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="DDPG", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_ddpg_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="DDPG", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)

def ppo_sac_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="SAC", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_sac_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="SAC", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_sac_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="SAC", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)

def ppo_td3_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="TD3", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_td3_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="TD3", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_td3_platform(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="TD3", env_name="Platform-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)


def ppo_ppo_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="PPO", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_ppo_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="PPO", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_ppo_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="PPO", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)

def ppo_a2c_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="A2C", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_a2c_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="A2C", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_a2c_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="A2C", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)

def ppo_ddpg_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="DDPG", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_ddpg_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="DDPG", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_ddpg_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="DDPG", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)

def ppo_sac_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="SAC", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_sac_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="SAC", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_sac_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="SAC", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)

def ppo_td3_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="PPO", continuousAlg="TD3", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def a2c_td3_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="A2C", continuousAlg="TD3", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
def dqn_td3_goal(max_steps, train_episodes, learning_steps, cycles, seeds, output_dir):
    run_converter(discreteAlg="DQN", continuousAlg="TD3", env_name="Goal-v0", max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seeds=seeds, use_tensorboard=False, write_csv=True, origin_log_dir=output_dir)
