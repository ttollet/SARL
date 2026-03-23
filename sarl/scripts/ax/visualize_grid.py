# %% Visualisation for fixed params grid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
os.getcwd()

# %% Configuration
csv_path = "runs/2026-03-17_03-50/grid_results.csv"

# %% Load and visualize data
df = pd.read_csv(csv_path)
print(df.head())

pivot = df.pivot(index='continuous_lr', columns='discrete_lr', values='mean_reward')
pivot = pivot.sort_index(ascending=False)
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", cbar_kws={'label': 'Mean Reward'})
plt.suptitle("Mean Reward: Discrete vs Continuous Learning Rate")
plt.title("160k Learning Steps per seed, 16 Cycles per seed; 15 seeds per trial")
plt.xlabel("Discrete Learning Rate")
plt.ylabel("Continuous Learning Rate")
plt.tight_layout()
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
plt.savefig(f"{timestamp_str}-heatmap.png")
plt.show()
