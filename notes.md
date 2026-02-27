Cycle:
discrete_agent.learn(n)
continuous_agent.learn(m)

SB3
- Handles episode stop/start logic internally
- Black box

SB3 Typically:
env = Cartpole()
agent = SB3Agent(..., env)
agent.learn(n)

Terminology:
- Optimisation Trials < Seeds < Cycle < Episodes < Timesteps
- 1 optimisation trial = avg mean_reward across n seeds per trial
  = Multiple agent training cycles per seed
  = Multiple episodes per cycle
  = Multiple timesteps per episode

Env timeline:
- Trial start
- Seed chosen
- 
- Env init with seed (sets speed)
- Agent init with env
- 
- Cycle start
- Agent.discrete learns
- Agent.continuous learns
- Cycle end
- Report mean_reward for wip learning
- Repeat for all cycles
-
- Seed end
- Report overall mean_reward to trial
-

DQN-PPO:
low | med | high
xxx
xxx
xxx
