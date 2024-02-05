import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm

from sarl.common.bester.agents.pdqn import PDQNAgent
from sarl.common.bester.common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from sarl.common.bester.environments.gym_goal.envs import GoalEnv
from sarl.common.bester.common.goal_domain import GoalObservationWrapper
from sarl.common.bester.common.goal_domain import GoalFlattenedActionWrapper
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.common.platform_domain import PlatformFlattenedActionWrapper


def _make_env(env_name: str, max_steps: int, seed: int):
    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env, deque_size=max_steps)  # Note stats
    env.seed(seed)  # Remove stochasticity
    np.random.seed(seed)
    return env


def _pad_action(act, act_param):
    params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
    params[act] = act_param
    return (act, params)


def _get_training_info(train_episodes, agent, env, max_steps, seed, pad_action, reward_scale=1):
    '''Train to output a list of returns by timestep.'''
    info_per_episode = []
    for _ in tqdm(range(train_episodes)):
        (observation, steps), info = env.reset(seed=seed)
        observation = np.array(observation, dtype=np.float32, copy=False)
        act, act_param, all_action_parameters = agent.act(observation)
        action = pad_action(act, act_param)

        # Episode loop
        agent.start_episode()
        reward = 0
        for j in tqdm(range(max_steps), desc=reward):
            (next_observation, steps), reward, terminated, truncated, info = env.step(action)
            next_observation = np.array(next_observation, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_observation)
            next_action = pad_action(next_act, next_act_param)

            r = reward * reward_scale
            agent.step(observation, (act, all_action_parameters), r, next_observation, (next_act, next_all_action_parameters), terminated or truncated, steps)

            # Reassign time-dependant variables
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            observation = next_observation

            if terminated or truncated:
                break
        agent.end_episode()
        info_per_episode.append(info)
    env.close()
    return info_per_episode


def test_pdqn_platform(train_episodes=2500, max_steps=250, seeds=[1]):
    '''Ensure P-DQN learns within Platform'''
    for seed in tqdm(seeds):
        # Environment initialisation
        env = _make_env(env_name="Platform-v0", max_steps=max_steps, seed=seed)

        # Setup
        ## Values
        initial_params_ = [3., 10., 400.]
        ## Scaled actions
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
        ## Initial weights
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        for a in range(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        ## Env Wrappers | TODO: Convert to environment options
        env = PlatformFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)  # Parameters -> [-1,1]
        env = ScaledStateWrapper(env)  # Observations -> [-1,1]

        ## Agent
        pdqn_setup = {  # TODO: Find & remove redundancy
            "observation_space": env.observation_space.spaces[0],
            "action_space": env.action_space,
            "learning_rate_actor": 0.001,  # 0.0001
            "learning_rate_actor_param": 0.00001,  # 0.001
            "epsilon_steps": 1000,
            "epsilon_final": 0.01,
            "gamma": 0.95,
            "clip_grad": 1,
            "indexed": False,
            "average": False,
            "random_weighted": False,
            "tau_actor": 0.1,
            "weighted": False,
            "tau_actor_param": 0.001,
            "initial_memory_threshold": 128,
            "use_ornstein_noise": True,
            "replay_memory_size": 20000,
            "inverting_gradients": True,
            "actor_kwargs": {'hidden_layers': (128,),
                             'action_input_layer': 0},
            "actor_param_kwargs": {
                'hidden_layers': (128,),
                'output_layer_init_std': 1e-4,
                'squashing_function': False},
            "zero_index_gradients": False,
            "seed": seed
        }
        agent = PDQNAgent(**pdqn_setup)
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)

        # Training | TODO: Simplify
        info_per_episode = _get_training_info(train_episodes, agent, env, max_steps, seed, _pad_action)

        # Check
        returns = [info["episode"]["r"] for info in info_per_episode]
        assert returns[-1] > returns[0]
    


def test_pdqn_goal(train_episodes=5000, max_steps=150, seeds=[1]):
    '''Ensure P-DQN learns within Goal'''
    for seed in seeds:

        # Environment initialisation
        env = _make_env(env_name="Goal-v0", max_steps=max_steps, seed=seed)

        # Setup
        ## Scaled actions (Unsure where the reasoning comes from for these)
        kickto_weights = np.array([[-0.375, 0.5, 0, 0.0625, 0],
                                   [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
        shoot_goal_left_weights = np.array([0.857346647646219686, 0])
        shoot_goal_right_weights = np.array([-0.857346647646219686, 0])
        ## Values
        initial_weights = np.zeros((4, 17))
        initial_weights[0, [10, 11, 14, 15]] = kickto_weights[0, 1:]
        initial_weights[1, [10, 11, 14, 15]] = kickto_weights[1, 1:]
        initial_weights[2, 16] = shoot_goal_left_weights[1]
        initial_weights[3, 16] = shoot_goal_right_weights[1]
        initial_bias = np.zeros((4,))
        initial_bias[0] = kickto_weights[0, 0]
        initial_bias[1] = kickto_weights[1, 0]
        initial_bias[2] = shoot_goal_left_weights[0]
        initial_bias[3] = shoot_goal_right_weights[0]
        reward_scale = 1./50.
        ## Env Wrappers | TODO: Convert to environment options
        env = GoalObservationWrapper(env)
        env = GoalFlattenedActionWrapper(env)
        env = ScaledParameterisedActionWrapper(env)  # Parameters -> [-1,1]
        env = ScaledStateWrapper(env)  # Observations -> [-1,1]

        ## Agent
        pdqn_setup = {  # TODO: Find & remove redundancy
            "observation_space": env.observation_space.spaces[0],
            "action_space": env.action_space,
            "learning_rate_actor": 0.001,  # 0.0001
            "learning_rate_actor_param": 0.00001,  # 0.001
            "epsilon_steps": 1000,
            "epsilon_final": 0.01,
            "gamma": 0.95,
            "clip_grad": 1,
            "indexed": False,
            "average": False,
            "random_weighted": False,
            "tau_actor": 0.1,
            "weighted": False,
            "tau_actor_param": 0.001,
            "initial_memory_threshold": 128,
            "use_ornstein_noise": True,
            "replay_memory_size": 20000,
            "inverting_gradients": True,
            "actor_kwargs": {'hidden_layers': (256,), 'output_layer_init_std': 1e-5, 'action_input_layer': 0},
            "actor_param_kwargs": {'hidden_layers': (256,), 'output_layer_init_std': 1e-5, 'squashing_function': False},
            "zero_index_gradients": False,
            "seed": seed
        }
        agent = PDQNAgent(**pdqn_setup)
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)

        # Training | TODO: Simplify
        info_per_episode = _get_training_info(train_episodes, agent, env, max_steps, seed, _pad_action, reward_scale)

        # Check
        returns = [info["episode"]["r"] for info in info_per_episode]
        assert returns[-1] > returns[0]
