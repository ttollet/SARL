- [x] Optimized hyperparameters -> full run chapter 1, vs. baselines
- [x] Optimization loss per trial
hyperparameters: best_scores_history.csv
loss per trial: client.csv

# HEC RUN
- Best hyperparameters for a given ALG1-ALG2-ENV combo
- PPO-PPO-Platform has an estimated best hyperparameters for discrete & cont., and also a baseline
## depends on 
- [x] SARL/sarl/scripts/ax/main.py runs without error
- [x] SARL/sarl/scripts/ax/main.py produces expected output still
- [ ] Git push locallly, then pull on HEC
- NB: Number of trials (`MAX_TRIALS`) & episodes per trials (`TRAIN_EPISODES`) are specified in config.py
- [ ] Decide choice of above constants
- [ ] HEC script can specify ALG1-ALG2-ENV combo

AFTER HEC
- [ ] Ensure train.py can specify hyp.params.
- [ ] Collect full runs of optimised hyp.params.
- [ ] Collect baselines

LOW PRIORITY
- [ ] Confirm max_trials is enforced on Ax
- [ ] Address Grid Search deprecation

