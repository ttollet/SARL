import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Soccer-v0',
    entry_point='sarl.common.bester.gym_soccer.envs:SoccerEnv',
    max_episode_steps=1000,  # (previously) timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='SoccerEmptyGoal-v0',
    entry_point='sarl.common.bester.gym_soccer.envs:SoccerEmptyGoalEnv',
    max_episode_steps=1000,  # timestep_limit=1000,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='SoccerScoreGoal-v0',
    entry_point='sarl.common.bester.gym_soccer.envs:SoccerScoreGoalEnv',
    max_episode_steps=1000,  # timestep_limit=1000,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='SoccerAgainstKeeper-v0',
    entry_point='sarl.common.bester.gym_soccer.envs:SoccerAgainstKeeperEnv',
    max_episode_steps=1000,  # timestep_limit=1000,
    reward_threshold=8.0,
    nondeterministic=True,
)
