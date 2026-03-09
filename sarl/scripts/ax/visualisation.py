# %% Visualisation for fixed params grid

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime

# %% Configuration
run_dir = "./runs/2026-03-04_16-19"
csv_files = [f for f in glob.glob(f"{run_dir}/*.csv") if "wip" not in f.lower()]


# %% Extract relevant data
def get_params_from_filename(f):
    return {key: float(v) for key, v in zip(["discrete_lr", "continuous_lr", "update_ratio"], f.split("/")[-1][:-4].split("-")[-1].split("_"))}

data = []
for f in csv_files:
    df = pd.read_csv(f)
    if len(df) > 0:
        param_dict = get_params_from_filename(f)
        # WARN: csv file reported parameters differs from used params in case of fixed params
        # TODO: fix implementation of fixed params
        data.append({
            'discrete_lr': param_dict['discrete_lr'],
            'continuous_lr': param_dict['continuous_lr'],
            'mean_reward': df['mean_reward'].iloc[0]
        })
df_all = pd.DataFrame(data)
df_all.head()

# %% Visualisation
pivot = df_all.pivot(index='continuous_lr', columns='discrete_lr', values='mean_reward')
pivot = pivot.sort_index(ascending=False)
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", cbar_kws={'label': 'Mean Reward'})
plt.title("Mean Reward: Discrete vs Continuous Learning Rate")
plt.xlabel("Discrete Learning Rate")
plt.ylabel("Continuous Learning Rate")
plt.tight_layout()
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
plt.savefig(f"{timestamp_str}-heatmap.png")
plt.show()
