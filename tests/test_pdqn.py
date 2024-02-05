import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from sarl.common.bester.agents.pdqn import PDQNAgent

from sarl.common.bester.common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper

from sarl.common.bester.environments.gym_goal.envs import GoalEnv
from sarl.common.bester.common.goal_domain import GoalObservationWrapper
from sarl.common.bester.common.goal_domain import GoalFlattenedActionWrapper

from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
# from sarl.common.bester.common.platform_domain import PlatformFlattenedActionWrapper


def test_pdqn_goal(train_episodes=1, max_steps=201, num_seeds=3):
    '''Ensure P-DQN learns within Goal'''
    for seed in range(num_seeds):
        np.random.seed(seed)

        # Environment initialisation
        env = gym.make("Goal-v0")
        env = RecordEpisodeStatistics(env, deque_size=max_steps)  # Note stats
        env.seed(seed)  # Remove stochasticity

        # Feature engineering | TODO: Convert to environment options
        env = GoalObservationWrapper(env)
        env = GoalFlattenedActionWrapper(env)
        env = ScaledStateWrapper(env)  # Observations -> [-1,1]
        env = ScaledParameterisedActionWrapper(env)  # Parameters -> [-1,1]

        # Setup: Ideally minimal
        reward_scale = 1./50.
        pdqn_setup = {
            # TODO: reduce redundant parts
            "observation_space": env.observation_space.spaces[0],
            "action_space": env.action_space,
            # learning_rate_actor=learning_rate_actor,  # 0.0001
            # learning_rate_actor_param=learning_rate_actor_param,  # 0.001
            # epsilon_steps=epsilon_steps,
            # epsilon_final=epsilon_final,
            # gamma=gamma,
            # clip_grad=clip_grad,
            # indexed=indexed,
            # average=average,
            # random_weighted=random_weighted,
            # tau_actor=tau_actor,
            # weighted=weighted,
            # tau_actor_param=tau_actor_param,
            # initial_memory_threshold=initial_memory_threshold,
            # use_ornstein_noise=use_ornstein_noise,
            # replay_memory_size=replay_memory_size,
            # inverting_gradients=inverting_gradients,
            "actor_kwargs": {'hidden_layers': (256,), 'output_layer_init_std': 1e-5, 'action_input_layer': 0},
            "actor_param_kwargs": {'hidden_layers': (256,), 'output_layer_init_std': 1e-5, 'squashing_function': False},
            # zero_index_gradients=zero_index_gradients,
            "seed": seed
        }
        agent = PDQNAgent(**pdqn_setup)

        # Training | TODO: Simplify
        def pad_action(act, act_param):
            params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
            params[act] = act_param
            return (act, params)
        info_per_episode = []

        for _ in range(train_episodes):
            (observation, steps), info = env.reset(seed=seed)
            observation = np.array(observation, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(observation)
            action = pad_action(act, act_param)

            # Episode loop
            agent.start_episode()
            for j in range(max_steps):
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

        # Check
        returns = [info["episode"]["r"] for info in info_per_episode]
        assert returns[-1] > 0.2


def test_pdqn_platform(train_episodes=1, max_steps=201, num_seeds=3):
    '''Ensure P-DQN learns within Platform'''
    raise NotImplementedError