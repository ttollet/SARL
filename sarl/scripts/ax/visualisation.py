# %% Visualisation for fixed params grid

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# %% Configuration
run_dir = "./runs/2026-03-04_16-19"
csv_files = [f for f in glob.glob(f"{run_dir}/*.csv") if "wip" not in f.lower()]
data = []

# %% Debug

# %% Extract relevant data
for f in csv_files:
    df = pd.read_csv(f)
    if len(df) > 0:
        data.append({
            'discrete_lr': df['discrete_learning_rate'].iloc[0],
            'continuous_lr': df['continuous_learning_rate'].iloc[0],
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
plt.show()
