from typing import Any

import numpy as np
from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ObsType
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList

from sarl.agents.callbacks.data_callback import DataCallback


class HybridPolicy:
    # For combining two separate policies for use with the converter
    # TODO: Consider inheriting from SB3 equivalent base class
    def __init__(self, discretePolicy=None, continuousPolicy=None,
        discreteAgent=None, continuousAgent=None, name=None, env_name=None,
        seed=None) -> None:
        self.agent = {key: None for key in ["discrete", "continuous"]}
        self.name = name
        self.timestep = None
        self.cycle = None
        self.env_name = env_name
        self.seed = seed

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

    def _getMeanReturn(self, eval_mdp, timesteps_per_cycle, cycle, eval_episodes):
        # timestep = timesteps_per_cycle * (cycle+1)
        timestep = self.timestep
        returns = []
        for i in range(eval_episodes):
            obs, info = eval_mdp.reset(seed=self.seed+cycle)
            # eval_mdp.seed(eval_mdp.)
            episode_over = False
            while not episode_over:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = eval_mdp.step(action)
                episode_over = terminated or truncated
            returns.append(info["episode"]["r"])
        return (timestep, np.mean(returns))


    def learn(self, total_timesteps, evaluation_interval=None,
        eval_mdp=None, cycles=1, callback=None, log_interval=1,
        tb_log_name='run', reset_num_timesteps=False, progress_bar=False,
        evaluation_episodes=15, log_dir=None):
        assert cycles >= 1
        if cycles > 1:
            assert total_timesteps % cycles == 0
        timesteps_per_agent = int(total_timesteps / cycles / len(self.agent.keys()))
        timesteps_per_cycle = timesteps_per_agent * 2
        self.timestep = 0
        for agent_type in self.agent.keys():
            self.agent[agent_type].agent_type = agent_type
            self.agent[agent_type].parent = self
        evaluation_returns = []
        for cycle in range(cycles-1):  # Was the '-1' necessary?
            self.cycle = cycle
            for agent_type in self.agent.keys():
                print(f"[{self.name}][Seed {self.agent[agent_type].seed}][Timestep {self.timestep}/{total_timesteps}][Cycle {cycle+1}/{cycles+1}][{agent_type}]: Learning for {timesteps_per_agent} timesteps...")
                # self.timestep = self.timestep + timesteps_per_agent
                agent = self.agent[agent_type]
                if agent is not None:
                    if isinstance(agent, BaseAlgorithm):
                        tb_log_name_for_component_agents = (tb_log_name+"_"+agent_type)
                        if callback is not None:
                            callback = CallbackList([callback])
                        self.agent[agent_type].learn(timesteps_per_agent,
                            callback, log_interval,
                            tb_log_name_for_component_agents,
                            reset_num_timesteps, progress_bar)
                    else:
                        raise NotImplementedError
            eval_bool = evaluation_interval is not None
            if eval_bool and (cycle + 1) % evaluation_interval == 0:
                mean_return = self._getMeanReturn(eval_mdp, timesteps_per_cycle,
                    cycle, evaluation_episodes)
                evaluation_returns.append(mean_return)
                file_name = f"{log_dir}/eval.csv"
                print(f"[REWARD]: Mean reward = {mean_return[1]}")
                print(f"[OUTPUT]: Writing to {file_name}")
                np.savetxt(fname=file_name, X=np.array(evaluation_returns),
                    header='"training_timesteps","mean__eval_episode_return"',
                    delimiter=',', fmt="%1.3f"
                )

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
            if self.combine_continuous_actions:  
                # ATTEMPT TO MAKE CONTINUOUS SPACE CONFORM TO SB3 PPO'S REQUIREMENTS
                self.action_parameter_indices_mapping = self.parent.action_parameter_indices_mapping
                self.action_space = self.parent.combine(self.action_space)  # Requires effort to restructure output later
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
                action = self.parent.uncombineAction(action)
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
        self.discrete_action_space = self.action_space[0]
        self.action_parameter_space = self.action_space[1]
        self.action_parameter_indices_mapping = self._getParamIndices()
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
            info = {}  # Resolves issue with RecordEpisodeStatistics wrapper
        else:
            if partial_action not in self.action_parameter_space:
                # Make it a tuple of arrays, assuming that's what the env wants
                indices = self.action_parameter_indices_mapping
                partial_action = tuple(np.array(partial_action[indices[action]]) for action in range(len(self.action_parameter_space)))
            assert partial_action in self.action_parameter_space
            # if partial_action not in self.action_parameter_space:
                # Assume combined and uncombine
                # partial_action = self.uncombine(self.action_parameter_space, partial_action)
            action = (np.int64(self.discrete_action_choice), partial_action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = (-1, obs)
        if info is None:
            info = {}
        step_output = obs, reward, terminated, truncated, info
        self.previous_step_output = {key: val for (key, val) in list(zip(STEP_KEYS, step_output))}
        return step_output

    def uncombineAction(self, action):
        # Return expected Tuple of boxes by partioning the box accordingly
        output = [np.empty(shape=box.shape) for box in self.action_parameter_space.spaces]
        for i in range(len(self.action_parameter_space)):
            output[i] = np.array(action[self.action_parameter_indices_mapping[i]])
        assert tuple(output) in self.action_parameter_space
        return tuple(output)
        
    def combine(self, space):
        # Receives a tuple of boxes
        # Outputs a single box
        # Samples of output box can be interpreted in terms of original tuple
        return spaces.Box(low=np.array(self.param_lows), high=np.array(self.param_highs), dtype=np.float32)

    def _getParamIndices(self):
        indices = {a: [] for a in list(range(self.discrete_action_space.n))}
        space = self.action_parameter_space
        self.param_lows = []
        self.param_highs = []
        for action in range(len(space)):
            box = space[action]
            for j in range(box.shape[0]):
                indices[action].append(len(self.param_highs))  # partition box based on original Tuple
                self.param_highs.append(box.high[j])
                self.param_lows.append(box.low[j])
            indices[action] = np.array(indices[action])
        return indices