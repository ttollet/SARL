from logging import info
import math

import numpy as np
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
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
    if env.spec.id == 'Platform-v0':
        return QPAMDPScaledParameterisedActionWrapper(ScaledStateWrapper(env))
    elif env.spec.id == 'Goal-v0':
        env = GoalObservationWrapper(env)
        return QPAMDPScaledParameterisedActionWrapper(ScaledStateWrapper(env))
    else:
        raise ValueError("Unsupported environment type")


def _make_env(env_name: str, max_steps: int, seed: int):
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env, deque_size=max_steps)  # Note stats
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)
    env = _wrap_for_qpamdp(env)
    return env


def _evaluate(eval_env, evaluation_returns, eval_episodes, log_dir, timestep, seed, agent):
    returns = []
    for i in range(eval_episodes):
        (obs, steps), info = eval_env.reset(seed=seed+timestep+i)
        episode_over = False
        while not episode_over:
            obs = np.array(obs, dtype=np.float32, copy=False)
            # action, act_param, all_action_parameters = agent.act(obs)
            action = agent.act(obs)
            (obs, steps), reward, terminated, truncated, info = eval_env.step(action)
            episode_over = terminated or truncated
        returns.append(info["episode"]["r"])
    mean_return = (timestep, np.mean(returns))
    evaluation_returns.append(mean_return)
    file_name = f"{log_dir}/eval.csv"
    print(f"[REWARD]: Mean reward = {mean_return[1]}")
    print(f"[OUTPUT]: Writing to {file_name}")
    np.savetxt(fname=file_name, X=np.array(evaluation_returns),
        header='"training_timesteps","mean_eval_episode_return"',
        delimiter=',', fmt="%1.3f"
    )
    return evaluation_returns


def _get_training_info(train_episodes, agent, env, max_steps, seed, output_dir, eval_env, eval_episodes, learning_steps):
    # NB train_episodes redundant, replaced with learning_steps
    if eval_episodes is None:
        eval_episodes = 15
    info_per_episode = []
    evaluation_returns = []
    eps_between_evals = 500
    # evaluations = math.ceil(train_episodes / eps_between_evals)

    # Initialize the agent
    new_info = agent.learn(env, eps_between_evals, max_steps)
    info_per_episode = info_per_episode + new_info
    training_timesteps = agent.training_timesteps
    evaluation_returns = _evaluate(eval_env, evaluation_returns, eval_episodes, output_dir, training_timesteps, seed, agent)

    # Complete the training loop
    # for i in range(evaluations-1):
    while agent.training_timesteps < learning_steps:
        new_info = agent.learn(env, eps_between_evals, max_steps, resume=True)  # Ensure train_eps >> initial_action_learning_episodes
        info_per_episode = info_per_episode + new_info
        training_timesteps = agent.training_timesteps
        evaluation_returns = _evaluate(eval_env, evaluation_returns, eval_episodes, output_dir, training_timesteps, seed, agent)
        # if agent.training_timesteps > learning_steps:
        #     break
    returns = [info["episode"]["r"] for info in info_per_episode]
    return returns


def qpamdp_platform(train_episodes=20, max_steps=201, seeds=[1], output_dir=None, learning_steps=None, cycles=None, eval_episodes=None):
    '''Q-PAMDP agent learns Platform'''
    if len(seeds) > 1:
        raise ValueError("Only one seed is supported per QPAMDP call")
    for seed in tqdm(seeds):
        # Env init
        env = _make_env(env_name="Platform-v0", max_steps=max_steps, seed=seed)
        eval_env = _make_env(env_name="Platform-v0", max_steps=max_steps, seed=seed+1)

        # Setup
        initial_params = [3., 10., 400.]
        ## Scaling
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
                            print_freq=100,
                            # Specified to avoid errors:
                            initial_action_learning_episodes=100,
                            parameter_rollouts=40,
                            parameter_updates=10,
                            action_relearn_episodes=50
                            )
        for a in range(env.action_space.spaces[0].n):
            agent.parameter_weights[a][0, 0] = initial_params[a]

        # Training
        returns = _get_training_info(train_episodes, agent, env, max_steps, seed, output_dir=output_dir, eval_env=eval_env, eval_episodes=eval_episodes, learning_steps=learning_steps)
        # info_per_episode = agent.learn(env, train_episodes, max_steps)
        # returns = [info["episode"]["r"] for info in info_per_episode]
        env.close()
        eval_env.close()
        return returns


def qpamdp_goal(train_episodes=4000, max_steps=150, seeds=[1], output_dir=None, learning_steps=None, cycles=None, eval_episodes=None):
    '''Q-PAMDP agent learns Goal'''
    if len(seeds) > 1:
        raise ValueError("Only one seed is supported per QPAMDP call")
    for seed in tqdm(seeds):
        # Env init
        env = _make_env(env_name="Goal-v0", max_steps=max_steps, seed=seed)
        eval_env = _make_env(env_name="Goal-v0", max_steps=max_steps, seed=seed+1)

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

        # Training
        returns = _get_training_info(train_episodes, agent, env, max_steps, seed, output_dir=output_dir, eval_env=eval_env, eval_episodes=eval_episodes, learning_steps=learning_steps)
        # info_per_episode = agent.learn(env, train_episodes, max_steps)
        # returns = [info["episode"]["r"] for info in info_per_episode]
        env.close()
        eval_env.close()
        return returns
