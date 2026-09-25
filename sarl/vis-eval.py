"""
Usage:
uv run python3 vis-eval.py /path/to/eval1.csv /path/to/eval2.csv ...
"""

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser(prog='vis-eval', description='Visualise eval.csv files.')
parser.add_argument('filepaths', nargs='+')
args = parser.parse_args()


def confidence_interval(reward, timesteps, confidence=1.96):
    """Compute mean reward and a confidence band at each unique timestep."""
    unique_ts = np.unique(timesteps)
    mean = np.array([reward[timesteps == t].mean() for t in unique_ts])
    sem = np.array([reward[timesteps == t].std(ddof=1) / np.sqrt((timesteps == t).sum()) for t in unique_ts])
    margin = confidence * sem
    return unique_ts, mean, margin


fig, ax = plt.subplots()
for fp in args.filepaths:
    parent_dir = fp.split('/')[-2]

    timesteps, reward, seed = np.loadtxt(fp, delimiter=',', unpack=True, skiprows=1)
    # ax.plot(timesteps, reward, label=parent_dir)
    unique_ts, mean, margin = confidence_interval(reward, timesteps)
    line, = ax.plot(unique_ts, mean, label=parent_dir)
    ax.fill_between(unique_ts, mean - margin, mean + margin, alpha=0.2, color=line.get_color())

ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean Reward")
ax.set_title("Baseline vs Optimised Hyperparameters")
plt.legend()
plt.show()
