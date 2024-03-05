#GOAL
self.action_space = spaces.Tuple((
    spaces.Discrete(num_actions),  # actions
    spaces.Tuple(  # parameters
        tuple(spaces.Box(
            PARAMETERS_MIN[i], PARAMETERS_MAX[i], dtype=np.float32) for i in range(num_actions))
    )
))
self.observation_space = spaces.Tuple((
    # spaces.Box(low=0., high=1., shape=self.get_state().shape, dtype=np.float32),  # scaled states
    spaces.Box(low=LOW_VECTOR, high=HIGH_VECTOR,
                dtype=np.float32),  # unscaled states
    # internal time steps (200 limit is an estimate)
    spaces.Discrete(200),
))

#PLATFORM
self.action_space = spaces.Tuple((
    spaces.Discrete(num_actions),  # actions
    # spaces.Box(Constants.PARAMETERS_MIN, Constants.PARAMETERS_MAX, dtype=np.float32),  # parameters
    spaces.Tuple(  # parameters
        tuple(spaces.Box(low=np.array([Constants.PARAMETERS_MIN[i]]), high=np.array([Constants.PARAMETERS_MAX[i]]), dtype=np.float32)
                for i in range(num_actions))
    )
))
self.observation_space = spaces.Tuple((
    spaces.Box(low=0., high=1.,
                shape=self.get_state().shape, dtype=np.float32),
    spaces.Discrete(200),  # steps (200 limit is an estimate)
))