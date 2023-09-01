import gymnasium as gym
from sarl.common.bester.environments.gym_platform.envs import PlatformEnv
from sarl.common.bester.environments.gym_goal.envs import GoalEnv
# from sarl.common.bester.environments.gym_soccer.envs import SoccerEnv, SoccerEmptyGoalEnv, SoccerScoreGoalEnv, SoccerAgainstKeeperEnv

STEPS_PER_TEST = 2


def _test_environment(env_name: str, steps: int = STEPS_PER_TEST):
    env = gym.make(env_name)
    observation, info = env.reset()
    for _ in range(2):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    assert True


def test_platform():
    _test_environment(env_name="Platform-v0")


def test_goal():
    _test_environment(env_name="Goal-v0")


# TODO: Handle `pip install gym[soccer]` dependencies. Don't forget to uncomment import too
# def test_soccer():
#     _test_environment(env_name="Soccer-v0")
#     _test_environment(env_name="SoccerEmptyGoal-v0")
#     _test_environment(env_name="SoccerScoreGoal-v0")
#     _test_environment(env_name="SoccerAgainstKeeper-v0")
