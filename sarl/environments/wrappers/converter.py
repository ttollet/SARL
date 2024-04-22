import numpy as np
from gymnasium import Env, Wrapper, spaces

class HybridPolicy:
    # For combining two separate policies for use with the converter
    def __init__(self, discretePolicy=None, continuousPolicy=None, discreteAgent=None, continuousAgent=None) -> None:
        self.agent = {key: None for key in ["discrete", "continuous"]}

        if discretePolicy is not None:
            self.discretePolicy = discretePolicy
        elif discreteAgent is not None:
            self.agent["discrete"] = discreteAgent
            self.discretePolicy = discreteAgent.predict

        if continuousPolicy is not None:
            self.continuousPolicy = continuousPolicy
        elif continuousAgent is not None:
            self.agent["continuous"] = continuousAgent
            self.continuousPolicy = continuousAgent.predict

    def learn(self, total_timesteps: int):
        for agent_type in self.agent.keys():
            if self.agent[agent_type] is not None:
                self.agent[agent_type].learn(total_timesteps)

    def predict(self, obs):
        if obs[0]==0:
            return self.discretePolicy(obs[1])
        else:
            assert obs[0]==1
            return self.continuousPolicy(obs[1])
        

class PamdpToMdpView(Env):
    def __init__(self, parent: Env, action_space_is_discrete: bool, internal_policy=None) -> None:
        super().__init__()
        if action_space_is_discrete:
            self.action_space = parent.discrete_action_space
        else:
            self.action_space = parent.action_parameter_space
        self.observation_space = parent.observation_space[1]
        self.reward_range = parent.reward_range  # TODO: Amend in future
        self.spec = parent.spec
        self.metadata = parent.metadata
        self.np_random = parent.np_random

        if internal_policy == None:
            if action_space_is_discrete:
                self.internal_policy = self.action_parameter_space.sample
            else:
                self.internal_policy = self.action_parameter_space.sample
        else:
            self.internal_policy = internal_policy
        self.action_space_is_discrete = action_space_is_discrete
        self.parent = parent
        if action_space_is_discrete:
            assert parent.expectingDiscreteAction()
        else:
            assert not parent.expectingDiscreteAction()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.parent.step(action)
        return self.parent.step(self.internal_policy(obs))

    def reset(self):
        return self.parent.reset()
    
    def render(self):
        return self.parent.render()
    
    def close(self):
        return self.parent.close()



STEP_KEYS = ["obs", "reward", "terminated", "truncated", "info"]
class PamdpToMdp(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        original_observation_space = self.observation_space
        self.observation_space = spaces.Tuple((
            spaces.Discrete(2),  # Awaiting discrete-action or action-parameter indicator
            original_observation_space
        ))

        self.discrete_action_space = self.action_space[0]
        self.action_parameter_space = self.action_space[1]
        # self.action_space = 

        self.previous_step_output = {key: None for key in STEP_KEYS[1:]}
        self.previous_step_output[STEP_KEYS[0]] = [0, None]  # Start with discrete action
        self.discrete_action_choice = None


    def getComponentMdp(self, action_space_is_discrete: bool, internal_policy=None) -> Env:
        return PamdpToMdpView(self, action_space_is_discrete, internal_policy)


    def expectingDiscreteAction(self):
        return self.previous_step_output["obs"][0] == 0

    
    def reset(self):
        obs, info = super().reset()
        self.previous_step_output["obs"] = (0, obs)
        return (0, obs), info


    def step(self, partial_action):
        if self.expectingDiscreteAction():
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