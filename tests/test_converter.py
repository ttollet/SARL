import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics  # Note stats
from gymnasium import ObservationWrapper
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor

import time

from sarl.environments.wrappers.converter import PamdpToMdp, HybridPolicy
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.environments.gym_goal.envs import GoalEnv


NUM_TRIALS = 1


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


def _test_converter(discrete=None, max_steps:int=250, learning_steps:int=250*1, cycles=1, seed:int=42, log_results:bool=False,
                    discreteAlg="PPO", continuousAlg="PPO"):  # 1 sample each 2048 timesteps for PPO
    '''Hybrid policy class can support a discrete learner'''
    for env_name in ["Platform-v0", "Goal-v0"]:
        for _ in range(NUM_TRIALS):
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
            if log_results:
                if discrete:
                    log_dir = f"tests/test_output/discrete/{discreteAlg.lower()}/{env_name.lower()}/{str(learning_steps)}steps"
                elif discrete is False:
                    log_dir = f"tests/test_output/continuous/{continuousAlg.lower()}/{env_name.lower()}/{str(learning_steps)}steps"
                else:
                    log_dir_discrete = f"tests/test_output/hybrid/{discreteAlg.lower()}-{continuousAlg.lower()}-discrete/{env_name.lower()}/{str(learning_steps)}steps" 
                    log_dir_continuous = f"tests/test_output/hybrid/{discreteAlg.lower()}-{continuousAlg.lower()}-continuous/{env_name.lower()}/{str(learning_steps)}steps" 

            if discrete:
                def continuousPolicy(x): return mdp.action_parameter_space.sample()
                discreteActionMDP = mdp.getComponentMdp(action_space_is_discrete=True, internal_policy=continuousPolicy)
                discreteAgent = {
                    "PPO": PPO("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir),
                    "AC2": A2C("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir)
                }[discreteAlg]
                agent = HybridPolicy(discreteAgent=discreteAgent, continuousPolicy=continuousPolicy)
            elif discrete is False:
                def discretePolicy(x): return mdp.discrete_action_space.sample()
                continuousActionMDP = mdp.getComponentMdp(action_space_is_discrete=False, internal_policy=discretePolicy, combine_continuous_actions=True)  # TODO: Ensure combine_continuous_actions is as desired log_dir = f"tests/test_output/continuous/ppo/{env_name.lower()}/{str(learning_steps)}steps"
                continuousAgent = {
                    "PPO": PPO("MlpPolicy", continuousActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir),
                }[continuousAlg]
                agent = HybridPolicy(discretePolicy=discretePolicy, continuousAgent=continuousAgent)
            else:
                discreteActionMDP = mdp.getComponentMdp(action_space_is_discrete=True)#, internal_policy=continuousAgent.predict)
                continuousActionMDP = mdp.getComponentMdp(action_space_is_discrete=False, combine_continuous_actions=True)#, internal_policy=discreteAgent.predict)
                discreteAgent = {
                    "PPO": PPO("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_discrete),
                    "AC2": A2C("MlpPolicy", discreteActionMDP, verbose=1, seed=seed, tensorboard_log=log_dir_discrete)
                }[discreteAlg]
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


def test_converter_discrete(max_steps:int=250, learning_steps:int=250*1, seed:int=42):  # Previously 250*500 learning steps
    return _test_converter(discrete=True, max_steps=max_steps, learning_steps=learning_steps, seed=seed)


def test_converter_continuous(max_steps:int=250, learning_steps:int=250*1, seed:int=42):
    return _test_converter(discrete=False, max_steps=max_steps, learning_steps=learning_steps, seed=seed)


def test_converter_both(max_steps:int=250, learning_steps:int=250*2*3, cycles=3, seed:int=42):
    # learning_steps=250*2*3 is 53s on M1 MBP
    return _test_converter(max_steps=max_steps, learning_steps=learning_steps, cycles=cycles, seed=seed)

# def test_converter_parity():
#     '''Converter outputs same cumulative reward as pre-converted envs.'''
#     pass
