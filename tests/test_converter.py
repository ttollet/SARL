import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics  # Note stats

from sarl.wrappers.converter import PamdpToMdp
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.environments.gym_goal.envs import GoalEnv


def _make_env(env_name: str, max_steps: int, seed: int):
    env = gym.make(env_name)
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)
    return env


def test_converter_sample(max_steps:int=5, seed:int=42) -> None:
    '''Converter can be sampled from'''
    for env_name in ["Platform-v0", "Goal-v0"]:
        pamdp = _make_env(env_name=env_name, max_steps=max_steps, seed=seed)
        mdp = PamdpToMdp(pamdp)

        # For clarity
        discrete_action_space = mdp.action_space[0]
        action_parameter_space = mdp.action_space[1]

        mdp.reset()
        for i in range(max_steps):
            if i % 2 == 0:
                mdp.step(discrete_action_space.sample())
            else:
                mdp.step(action_parameter_space.sample())
    return None


def test_converter_parity():
    '''Converter outputs same cumulative reward'''
    pass