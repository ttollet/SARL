import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics  # Note stats

from sarl.environments.wrappers.converter import PamdpToMdp, HybridPolicy
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.environments.gym_goal.envs import GoalEnv


def _make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)
    return env


def test_converter_agent_sample(max_steps:int=5, seed:int=42) -> None:
    '''Hybrid policy class can correctly sample converter action space'''
    for env_name in ["Platform-v0", "Goal-v0"]:
        pamdp = _make_env(env_name=env_name, seed=seed)
        mdp = PamdpToMdp(pamdp)
        agent = HybridPolicy(
            discretePolicy = lambda x: mdp.discrete_action_space.sample(),
            continuousPolicy = lambda x: mdp.action_parameter_space.sample()
        )

        obs, info = mdp.reset()
        for i in range(max_steps*2):
            if mdp.expectingDiscreteAction():
                obs, reward, terminated, truncated, info = mdp.step(agent.act(obs))
            else:
                obs, reward, terminated, truncated, info = mdp.step(agent.act(obs))
    return None


def test_converter_discrete_learning(max_steps:int=500, seed:int=42) -> None:
    '''Hybrid policy class can correctly sample converter action space'''
    for env_name in ["Platform-v0", "Goal-v0"]:
        pamdp = _make_env(env_name=env_name, seed=seed)
        mdp = PamdpToMdp(pamdp)
        agent = HybridPolicy(
            discretePolicy = lambda x: mdp.discrete_action_space.sample(),
            continuousPolicy = lambda x: mdp.action_parameter_space.sample()
        )

        obs, info = mdp.reset()
        for i in range(max_steps*2):
            if mdp.expectingDiscreteAction():
                obs, reward, terminated, truncated, info = mdp.step(agent.act(obs))
            else:
                obs, reward, terminated, truncated, info = mdp.step(agent.act(obs))
    return None


def test_converter_parity():
    '''Converter outputs same cumulative reward'''
    pass