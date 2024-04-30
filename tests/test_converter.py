import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics  # Note stats
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO

import time

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


def test_converter_discrete_duration(max_steps:int=500, learning_steps=500, seed:int=42) -> None:
    '''Hybrid policy class can correctly sample converter action space'''
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
        continuousPolicy = lambda x: mdp.action_parameter_space.sample()
        discreteActionMDP = mdp.getComponentMdp(action_space_is_discrete=True, internal_policy=continuousPolicy)
        discreteAgent = PPO("MlpPolicy", discreteActionMDP, verbose=2, seed=seed, tensorboard_log="tests/test_output")
        agent = HybridPolicy(discreteAgent=discreteAgent, continuousPolicy=continuousPolicy)

        # Learning  TODO: Plot and fix
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
    return None
# test_converter_discrete_duration()

def test_converter_parity():
    '''Converter outputs same cumulative reward'''
    pass
