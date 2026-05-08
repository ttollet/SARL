# %% Imports
import numpy as np
# import pandas as pd
from scipy.stats import ttest_ind

# %% Load data
# results = pd.read_csv("")

# %% Perform t-test
# ttest_results = ttest_rel(results["mean_reward"], results["mean_reward_se"])
v1 = np.random.normal(loc=0, size=100)
v2 = np.random.normal(size=100)

res = ttest_ind(v1, v2)

print(res)
