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


**2025-03-05 - Post Raphael meeting

Have:
- 9 csv files
- Each has single value, due to fixed param implementation
Want:
- 3x3 grid -> creating notebook for this purpose
Have:
- 3x3 grid
Want:
- Dist of avg. rewards (see the seeds' individual rewards and variances plotted for a given optimisation trial)
- Line plot analogous to paper with std confidence interval sufficiently small


**2026-03-09 - Meeting Talking Points**

Recent Progress (Git Commits):
- Visualisation work: Updated visualisation.py for heatmaps (mean rewards across discrete/continuous LR grids)
- Grid preparation: 3x3 grid complete (9 CSV files with fixed param combinations)
- Ax driver improvements: Refined ax_driver.py output, fixed bugs (timesteps vs mean eval return, last cycle skipping)
- Data pipeline: Optimisation saves to CSV, resolved Goal domain codebase issues

Current Status:
- 3x3 grid complete - visualising mean rewards from optimised SB3 agents
- Next: 15-trial averaging - need distributions of avg rewards with std devs for each optimised combination

Chapter 2 Context:
- Thesis chapter 2: Using Bayesian optimisation via Ax to improve RL agent performance
- Test domains: Platform & Goal environments
- Baselines: QPAMDP and PDQN for comparison

Upcoming Tasks:
- Writing not started for chapter 2
- Verify optimisation converges to better hyperparameter choices
- Timeline for completing 15-trial evaluation

