from gymnasium.envs.registration import register

register(
    id='Goal-v0',
    entry_point='sarl.common.bester.environments.gym_goal.envs:GoalEnv',
)
