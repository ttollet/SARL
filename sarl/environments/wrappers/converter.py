from typing import Any

import numpy as np
from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ObsType
from stable_baselines3.common.base_class import BaseAlgorithm

class HybridPolicy:
    # For combining two separate policies for use with the converter
    # TODO: Consider inheriting from SB3 equivalent base class
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

    def learn(self, total_timesteps, cycles=1, callback=None, log_interval=1, tb_log_name='run', reset_num_timesteps=False, progress_bar=False):
        assert cycles >= 1
        if cycles > 1:
            assert total_timesteps % cycles == 0
        timesteps_per_agent = total_timesteps / cycles / len(self.agent.keys())
        for _ in range(cycles):
            for agent_type in self.agent.keys():
                agent = self.agent[agent_type]
                if agent is not None:
                    if isinstance(agent, BaseAlgorithm):
                        tb_log_name_for_component_agents = (tb_log_name+"_"+agent_type)
                        self.agent[agent_type].learn(timesteps_per_agent, callback, log_interval, 
                                                     tb_log_name_for_component_agents, reset_num_timesteps, 
                                                     progress_bar)  # TODO: Share timestep number
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
                prediction = int(prediction)
        else:
            prediction = policy(obs[1])
        return prediction

class PamdpToMdpView(Env):
    def __init__(self, parent:Env, action_space_is_discrete:bool, internal_policy=None, combine_continuous_actions:bool=False) -> None:
        """Initialise an MDP to provide a method of only taking either discrete actions or action-parameters, whilst an internal policy handles the unchosen component.

        Args:
            parent (Env): PAMDP which this artificial MDP interacts with.
            action_space_is_discrete (bool): Whether this MDP is intended for a discrete or continuous policy.
            combine_continuous_actions (bool, optional): To improve compatability. Defaults to True.
            internal_policy (_type_, optional): Specify policy to handle non-agent action component selection. Defaults to None (resulting in a random policy).
        """

        super().__init__()

        self.combine_continuous_actions = combine_continuous_actions
        self.observation_space = parent.observation_space[1]
        self.reward_range = parent.reward_range  # TODO: Amend in future
        self.spec = parent.spec
        self.metadata = parent.metadata
        self.np_random = parent.np_random
        self.parent = parent

        if action_space_is_discrete:
            self.action_space = parent.discrete_action_space
        else:
            self.action_space = parent.action_parameter_space
            if self.combine_continuous_actions:  # ATTEMPT TO MAKE CONTINUOUS SPACE CONFORM TO SB3 PPO'S REQUIREMENTS
                self.action_parameter_indices_mapping = {a: [] for a in list(range(self.parent.discrete_action_space.n))}
                # ^^^ Stores which of the single box's indicies correspond to which discrete actions
                self.action_space = self._combine(self.action_space)  # Requires effort to restructure output later

        if internal_policy is None:
            if action_space_is_discrete:
                self.internal_policy = lambda _: parent.action_parameter_space.sample()
            else:
                self.internal_policy = lambda _: parent.discrete_action_space.sample()
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
            if self.combine_continuous_actions:
                action = self._uncombine_action(action)
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

    def _combine(self, space):
        # Receives a tuple of boxes
        # Outputs a single box 
        # Samples of output box can be interpreted in terms of original tuple
        lows = []
        highs = []
        for action in range(len(space)):
            box = space[action]
            for j in range(box.shape[0]):
                self.action_parameter_indices_mapping[action].append(len(highs))  # partition box based on original Tuple
                highs.append(box.high[j])
                lows.append(box.low[j])
            self.action_parameter_indices_mapping[action] = np.array(self.action_parameter_indices_mapping[action])
        self.parent.action_parameter_indices_mapping = self.action_parameter_indices_mapping
        return spaces.Box(low=np.array(lows), high=np.array(highs), dtype=np.float32)

    def _uncombine_action(self, action):
        # Return expected Tuple of boxes by partioning the box accordingly
        output = [np.empty(shape=box.shape) for box in self.parent.action_parameter_space.spaces]
        for i in range(len(self.parent.action_parameter_space)):
            output[i] = np.array(action[self.action_parameter_indices_mapping[i]])
        assert tuple(output) in self.parent.action_parameter_space
        return tuple(output)


STEP_KEYS = ["obs", "reward", "terminated", "truncated", "info"]
class PamdpToMdp(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        self.discrete_action_space = self.action_space[0]
        self.action_parameter_space = self.action_space[1]
        self.action_parameter_indices_mapping: dict
        # self.action_space = 

        original_observation_space = self.observation_space
        self.observation_space = spaces.Tuple((
            spaces.Discrete(2),  # Awaiting discrete-action or action-parameter indicator
            original_observation_space
        ))

        self.previous_step_output = {key: None for key in STEP_KEYS[1:]}
        self.previous_step_output[STEP_KEYS[0]] = [-1, None]  # Start with discrete action
        self.discrete_action_choice = None


    def getComponentMdp(self, action_space_is_discrete: bool, internal_policy=None, combine_continuous_actions:bool=False) -> Env:
        return PamdpToMdpView(self, action_space_is_discrete, internal_policy, combine_continuous_actions=combine_continuous_actions)


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
            if partial_action not in self.action_parameter_space:
                # Make it a tuple of arrays, assuming that's what the env wants
                indices = self.action_parameter_indices_mapping
                partial_action = tuple(np.array(partial_action[indices[action]]) for action in range(len(self.action_parameter_space)))
            assert partial_action in self.action_parameter_space
            # if partial_action not in self.action_parameter_space:
                # Assume combined and uncombine
                # partial_action = self._uncombine(self.action_parameter_space, partial_action)
            action = (np.int64(self.discrete_action_choice), partial_action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = (-1, obs)
        step_output = obs, reward, terminated, truncated, info
        self.previous_step_output = {key: val for (key, val) in list(zip(STEP_KEYS, step_output))}
        return step_output
