import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm

from sarl.common.bester.agents.qpamdp import QPAMDPAgent
from sarl.common.bester.agents.sarsa_lambda import SarsaLambdaAgent
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.environments.gym_goal.envs import GoalEnv
from sarl.common.bester.common.wrappers import QPAMDPScaledParameterisedActionWrapper
from sarl.common.bester.common.wrappers import ScaledStateWrapper
from sarl.common.bester.common.goal_domain import CustomFourierBasis, GoalObservationWrapper
from sarl.common.bester.environments.gym_goal.envs.config import GOAL_WIDTH, PITCH_WIDTH, PITCH_LENGTH


def _wrap_for_qpamdp(env):
    return QPAMDPScaledParameterisedActionWrapper(ScaledStateWrapper(env))


def test_qpamdp_platform(train_eps=20, max_steps=201, seeds=[1]):
    '''Ensure Q-PAMDP learns within Platform'''
    for seed in seeds:
        # Env init
        env = gym.make("Platform-v0")
        env = RecordEpisodeStatistics(env, deque_size=max_steps)
        env.seed(seed)
        np.random.seed(seed)

        # Setup
        initial_params = [3., 10., 400.]
        ## Scaling
        env = _wrap_for_qpamdp(env)
        variances = [0.0001, 0.0001, 0.0001]
        for a in range(env.action_space.spaces[0].n):
            initial_params[a] = 2. * (initial_params[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
        alpha_param = 0.1
        ## Misc
        act_obs_index = [action_index for action_index, _ in enumerate(["self.player.position[0]",
                                                                        "self.player.velocity[0]",
                                                                        "enemy.position[0]",
                                                                        "enemy.dx"])]
        
        # Agent init
        discrete_agent = SarsaLambdaAgent(env.observation_space.spaces[0],
                                          env.action_space.spaces[0],
                                          alpha=alpha_param,
                                          gamma=0.999,
                                          lmbda=0.5,
                                          order=6,
                                          seed=seed,
                                          observation_index=act_obs_index,
                                          gamma_step_adjust=True)
        agent = QPAMDPAgent(observation_space=env.observation_space.spaces[0],
                            action_space=env.action_space,
                            seed=seed,
                            action_obs_index=act_obs_index,
                            variances=variances,  # TODO: How did Masson decide these?
                            discrete_agent=discrete_agent,
                            print_freq=100
                            )
        for a in range(env.action_space.spaces[0].n):
            agent.parameter_weights[a][0, 0] = initial_params[a]

        # Training
        info_per_episode = agent.learn(env, train_eps, max_steps)
        returns = [info["episode"]["r"] for info in info_per_episode]
        # print("Ave. return =", sum(returns) / len(returns))
        # print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
        env.close()
        assert returns[0] < returns[-1]
        # assert returns[-1] > 0.2


def test_qpamdp_goal(train_eps=4000, max_steps=150, seeds=[1]):
    '''Ensure Q-PAMDP learns within Goal'''
    for seed in seeds:
        # Env init
        env = gym.make("Goal-v0")
        env = RecordEpisodeStatistics(env, deque_size=max_steps)
        env.seed(seed)
        np.random.seed(seed)


        # Setup
        ## Values
        variances = [0.01, 0.01, 0.01]
        xfear = 50.0 / PITCH_LENGTH
        yfear = 50.0 / PITCH_WIDTH
        caution = 5.0 / PITCH_WIDTH
        kickto_weights = np.array([[2.5, 1, 0, xfear, 0], [0, 0, 1 - caution, 0, yfear]])
        initial_parameter_weights = [
            kickto_weights,
            np.array([[GOAL_WIDTH / 2 - 1, 0]]),
            np.array([[-GOAL_WIDTH / 2 + 1, 0]])
        ]

        ## Env wrappers
        env = GoalObservationWrapper(env)
        env = _wrap_for_qpamdp(env)
        ### Scaling
        variances[0] = 0.0001
        variances[1] = 0.0001
        variances[2] = 0.0001
        alpha_param = 0.06
        initial_parameter_weights[0] = np.array([[-0.375, 0.5, 0, 0.0625, 0],
                                [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
        initial_parameter_weights[1] = np.array([0.857346647646219686, 0])
        initial_parameter_weights[2] = np.array([-0.857346647646219686, 0])

        ## Components
        action_obs_index = np.arange(14)
        param_obs_index = np.array([
            np.array([10, 11, 14, 15]),  # ball_features
            np.array([16, 0, 0, 0]),  # keeper_features  # Padded w/ zeroes
            np.array([16, 0, 0, 0]),  # keeper_features  # to stop error
        ])
        basis = CustomFourierBasis(14, env.observation_space.spaces[0].low[:14], env.observation_space.spaces[0].high[:14])
        discrete_agent = SarsaLambdaAgent(env.observation_space.spaces[0], env.action_space.spaces[0], basis=basis, seed=seed, alpha=0.01,
                                        lmbda=0.1, gamma=0.9, temperature=1.0, cooling=1.0, scale_alpha=False,
                                        use_softmax=True,
                                        observation_index=action_obs_index, gamma_step_adjust=False)


        # Agent init
        agent = QPAMDPAgent(env.observation_space.spaces[0], env.action_space, alpha=alpha_param, initial_action_learning_episodes=4000,
                            seed=seed, action_obs_index=action_obs_index, parameter_obs_index=param_obs_index,
                            variances=variances, discrete_agent=discrete_agent, action_relearn_episodes=2000,
                            parameter_updates=1000, parameter_rollouts=50, norm_grad=True, print_freq=100,
                            phi0_func=lambda state: np.array([1, state[1], state[1]**2]),
                            phi0_size=3)
        # agent = QPAMDPAgent(observation_space=env.observation_space.spaces[0],
        #                     action_space=env.action_space,
        #                     seed=seed,
        #                     print_freq=100
        #                     )

        # Training
        info_per_episode = agent.learn(env, train_eps, max_steps)
        returns = [info["episode"]["r"] for info in info_per_episode]
        env.close()
        assert returns[0] < returns[-1]
