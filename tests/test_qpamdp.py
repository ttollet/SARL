import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from sarl.common.bester.agents.qpamdp import QPAMDPAgent
from sarl.common.bester.agents.sarsa_lambda import SarsaLambdaAgent
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.environments.gym_goal.envs import GoalEnv
from sarl.common.bester.common.wrappers import QPAMDPScaledParameterisedActionWrapper
from sarl.common.bester.common.wrappers import ScaledStateWrapper


# '''
# def _run_agent_on_env(agent, env, steps=2):
#     (state, _), info = env.reset()
#     for _ in range(steps):
#         action = agent.act(state)  # this is where you would insert your policy
#         (state, _), reward, terminated, truncated, info = env.step(action)

#         if terminated or truncated:
#             (state, _), info = env.reset()
#     env.close()
# '''


# '''
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.base_class import BaseAlgorithm
# def _test_agent_on_env_sb3_style(env_name: str, algorithm: BaseAlgorithm, policy_name: str, save_name: str):
#     # Based on example at https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#basic-usage-training-saving-loading

#     # Create environment
#     env = gym.make(env_name)

#     # Instantiate the agent
#     model = algorithm(policy_name, env, verbose=1)
#     # Train the agent and display a progress bar
#     model.learn(total_timesteps=int(2e5), progress_bar=True)
#     # Save the agent
#     model.save(save_name)
#     del model  # delete trained model to demonstrate loading

#     # Load the trained agent
#     # NOTE: if you have loading issue, you can pass `print_system_info=True`
#     # to compare the system on which the model was trained vs the current one
#     # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
#     model = algorithm.load(save_name, env=env)

#     # Evaluate the agent
#     # NOTE: If you use wrappers with your environment that modify rewards,
#     #       this will be reflected here. To evaluate with original rewards,
#     #       wrap environment in a "Monitor" wrapper before other wrappers.
#     mean_reward, std_reward = evaluate_policy(
#         model, model.get_env(), n_eval_episodes=10)

#     # Enjoy trained agent
#     vec_env = model.get_env()
#     obs = vec_env.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, dones, info = vec_env.step(action)
#         # vec_env.render("human")
# '''


def test_qpamdp_platform(train_eps=20, max_steps=201, scale=True):
    '''Ensure Q-PAMDP learns within Platform'''
    for seed in [1, 2, 3]:
        # Env init
        env = gym.make("Platform-v0")
        env = RecordEpisodeStatistics(env, deque_size=max_steps)
        env.seed(seed)
        np.random.seed(seed)

        # Setup
        initial_params = [3., 10., 400.]
        if scale:
            # Is this important?
            env = ScaledStateWrapper(env)
            variances = [0.0001, 0.0001, 0.0001]
            for a in range(env.action_space.spaces[0].n):
                initial_params[a] = 2. * (initial_params[a] - env.action_space.spaces[1].spaces[a].low) / (
                            env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
            env = QPAMDPScaledParameterisedActionWrapper(env)
            alpha_param = 0.1
        else:
            # This case doesn't learn well?
            variances = [0.1, 0.1, 0.01]
            alpha_param = 1.0
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


# def test_qpamdp_goal(train_eps=1, max_steps=2):
#     '''Ensure Q-PAMDP learns within Goal'''
#     for seed in [1, 2, 3]:
#         # Env init
#         env = gym.make("Goal-v0")
#         env = RecordEpisodeStatistics(env, deque_size=max_steps)
#         env.seed(seed)
#         np.random.seed(seed)

#         # Agent init
#         agent = QPAMDPAgent(observation_space=env.observation_space.spaces[0],
#                             action_space=env.action_space,
#                             seed=seed,
#                             print_freq=100
#                             )

#         # Training
#         info_per_episode = agent.learn(env, train_eps, max_steps)
#         returns = [info["episode"]["r"] for info in info_per_episode]
#         env.close()
#         assert returns[0] < returns[-1]
