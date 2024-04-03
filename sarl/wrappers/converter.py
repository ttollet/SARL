import numpy as np
from gymnasium import Env, Wrapper, spaces

STEP_KEYS = ["obs", "reward", "terminated", "truncated", "info"]

class PamdpToMdp(Wrapper):
    def _expectingDiscreteAction(self):
        return self.previous_step_output["obs"][0] == 0

    def __init__(self, env: Env):
        super().__init__(env)

        # Redefine attributes
        def augmentedObservationSpace(original_observation_space):
            augmented_observation_space = spaces.Tuple((
                spaces.Discrete(2),  # Awaiting discrete-action or action-parameter indicator
                original_observation_space
            ))
            return augmented_observation_space
        self.observation_space = augmentedObservationSpace(self.observation_space)

        # Track previous observation
        self.previous_step_output = {key: (lambda x: (0, None) if key=="obs" else None)(key) for key in STEP_KEYS}
        self.discrete_action_choice = None


    def step(self, partial_action):
        if self._expectingDiscreteAction():
            assert type(partial_action) is np.int64
            self.discrete_action_choice = partial_action
            obs = (1, self.previous_step_output["obs"][1])
            reward = 0
            terminated, truncated, info = (self.previous_step_output[key] for key in STEP_KEYS[2:])
        else:
            assert type(partial_action) is tuple
            action = (self.discrete_action_choice, partial_action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = (0, obs)
        step_output = obs, reward, terminated, truncated, info
        self.previous_step_output = {key: val for (key, val) in list(zip(STEP_KEYS, step_output))}
        return step_output