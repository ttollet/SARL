from typing import Any

import numpy as np
from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ObsType
from gymnasium.spaces.utils import flatten_space
from stable_baselines3.common.base_class import BaseAlgorithm

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

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name='run', reset_num_timesteps=True, progress_bar=False):
        for agent_type in self.agent.keys():
            agent = self.agent[agent_type]
            if agent is not None:
                if isinstance(agent, BaseAlgorithm):
                    self.agent[agent_type].learn(total_timesteps, callback, log_interval, (tb_log_name+"_"+agent_type), reset_num_timesteps, progress_bar)  # TODO: Share timestep number
                else:
                    raise NotImplementedError

    def predict(self, obs):
        if obs[0]==-1:
            policy = self.discretePolicy
        else:
            assert obs[0] > -1
            policy = self.continuousPolicy
        if policy.__qualname__ == "BaseAlgorithm.predict":  # If the policy is the method of a StableBaselines3 BaseAlgorithm object
            prediction = policy(obs[1])[0]  # Since prediction[1] is irrelevant unused hidden state information
            if obs[0]==-1:
                return int(prediction)
            else:
                return prediction
        else:
            return policy(obs[1])

class PamdpToMdpView(Env):
    def __init__(self, parent:Env, action_space_is_discrete:bool, flatten_continuous_actions:bool=True, internal_policy=None) -> None:
        """Initialise an MDP to provide a method of only taking either discrete actions or action-parameters, whilst an internal policy handles the unchosen component.

        Args:
            parent (Env): PAMDP which this artificial MDP interacts with.
            action_space_is_discrete (bool): Whether this MDP is intended for a discrete or continuous policy.
            flatten_continuous_actions (bool, optional): To improve compatability. Defaults to True.
            internal_policy (_type_, optional): Specify policy to handle non-agent action component selection. Defaults to None (resulting in a random policy).
        """

        super().__init__()
        if action_space_is_discrete:
            self.action_space = parent.discrete_action_space
        else:
            self.action_space = parent.action_parameter_space
            if flatten_continuous_actions:  # ATTEMPT TO MAKE CONTINUOUS SPACE CONFORM TO SB3 PPO'S REQUIREMENTS
                self.action_space = flatten_space(self.action_space)  # Requires effort to restructure output later
        self.observation_space = parent.observation_space[1]
        self.reward_range = parent.reward_range  # TODO: Amend in future
        self.spec = parent.spec
        self.metadata = parent.metadata
        self.np_random = parent.np_random
        self.parent = parent

        if internal_policy == None:
            if action_space_is_discrete:
                self.internal_policy = lambda obs: parent.action_parameter_space.sample()
            else:
                self.internal_policy = lambda obs: parent.discrete_action_space.sample()
        else:
            self.internal_policy = internal_policy
        self.action_space_is_discrete = action_space_is_discrete
        if action_space_is_discrete:
            assert parent.expectingDiscreteAction()
        else:
            view_obs = self.parent.previous_step_output["obs"][1]
            obs, reward, terminated, truncated, info = self.parent.step(self.internal_policy(view_obs))
            assert not parent.expectingDiscreteAction()

    def step(self, action):
        if self.action_space_is_discrete:
            obs, reward, terminated, truncated, info = self.parent.step(action)
            view_obs = obs[1]
            obs, reward, terminated, truncated, info = self.parent.step(self.internal_policy(view_obs))
            view_obs = obs[1]
        else:
            view_obs = self.parent.previous_step_output["obs"][1]
            obs, reward, terminated, truncated, info = self.parent.step(self.internal_policy(view_obs))
            view_obs = obs[1]
            obs, reward, terminated, truncated, info = self.parent.step(action)
            view_obs = obs[1]
            
        return view_obs, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.parent.reset(seed=seed, options=options)
        view_obs = obs[0] if self.action_space_is_discrete else obs[1]
        return view_obs, info
    
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
        self.previous_step_output[STEP_KEYS[0]] = [-1, None]  # Start with discrete action
        self.discrete_action_choice = None


    def getComponentMdp(self, action_space_is_discrete: bool, internal_policy=None) -> Env:
        return PamdpToMdpView(self, action_space_is_discrete, internal_policy)


    def expectingDiscreteAction(self):
        assert -1 not in self.discrete_action_space
        return self.previous_step_output["obs"][0] == -1  # Since discrete actions are indicated >=0

    
    def reset(self, *, seed = None, options = None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        converted_obs = (-1, obs)  # Indicator, obs
        self.previous_step_output["obs"] = converted_obs
        return converted_obs, info


    def step(self, partial_action):
        if self.expectingDiscreteAction():
            assert partial_action in self.discrete_action_space
            self.discrete_action_choice = partial_action
            obs = (partial_action, self.previous_step_output["obs"][1])
            reward = 0
            terminated, truncated, info = (self.previous_step_output[key] for key in STEP_KEYS[2:])
        else:
            assert partial_action in self.action_parameter_space
            action = (self.discrete_action_choice, partial_action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = (-1, obs)
        step_output = obs, reward, terminated, truncated, info
        self.previous_step_output = {key: val for (key, val) in list(zip(STEP_KEYS, step_output))}
        return step_output