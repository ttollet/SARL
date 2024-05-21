import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics  # Note stats
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import time

from sarl.environments.wrappers.converter import PamdpToMdp, HybridPolicy
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.environments.gym_goal.envs import GoalEnv



def _make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)
    return env


def test_converter_agent_sample(max_steps:int=5, seed:int=42):
    '''Hybrid policy class can correctly sample converter action space'''
    for env_name in ["Platform-v0", "Goal-v0"]:
        pamdp = _make_env(env_name=env_name, seed=seed)
        mdp = PamdpToMdp(pamdp)

        # Agent setup
        agent = HybridPolicy(
            discretePolicy = lambda x: mdp.discrete_action_space.sample(),
            continuousPolicy = lambda x: mdp.action_parameter_space.sample()
        )

        obs, info = mdp.reset()
        for i in range(max_steps*2):
            if mdp.expectingDiscreteAction():
                obs, reward, terminated, truncated, info = mdp.step(agent.predict(obs))
            else:
                obs, reward, terminated, truncated, info = mdp.step(agent.predict(obs))
    return None


def _test_converter_duration(discrete=None, max_steps:int=250, learning_steps:int=250*1000, seed:int=42):  # 1 sample each 2048 timesteps for PPO
    '''Hybrid policy class can support a discrete learner'''
    for env_name in ["Platform-v0", "Goal-v0"]:
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
        if discrete==True:
            continuousPolicy = lambda x: mdp.action_parameter_space.sample()
            discreteActionMDP = mdp.getComponentMdp(action_space_is_discrete=True, internal_policy=continuousPolicy)
            discreteAgent = PPO("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=("tests/test_output/discrete_learning/ppo_"+env_name.lower()))
            agent = HybridPolicy(discreteAgent=discreteAgent, continuousPolicy=continuousPolicy)
        else:
            discretePolicy = lambda x: mdp.discrete_action_space.sample()
            continuousActionMDP = mdp.getComponentMdp(action_space_is_discrete=False, internal_policy=discretePolicy)
            continuousAgent = PPO("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=("tests/test_output/continuous_learning/ppo_"+env_name.lower()))
            agent = HybridPolicy(discretePolicy=discretePolicy, continuousAgent=continuousAgent)

        agent.learn(learning_steps, progress_bar=True)

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


def test_converter_discrete_duration(max_steps:int=250, learning_steps:int=250*500, seed:int=42):
    return _test_converter_duration(discrete=True, max_steps=max_steps, learning_steps=learning_steps, seed=seed)


def test_converter_continuous_duration(max_steps:int=250, learning_steps:int=250*500, seed:int=42):
    return _test_converter_duration(discrete=False, max_steps=max_steps, learning_steps=learning_steps, seed=seed)


# def test_converter_parity():
#     '''Converter outputs same cumulative reward'''
#     pass
