"""
Usage:
uv run python3 vis-eval.py /path/to/eval.csv
"""

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser(prog='vis-eval', description='Visualise eval.csv files.')
parser.add_argument('filepaths', nargs='+')
args = parser.parse_args()

fig, ax = plt.subplots()
for fp in args.filepaths:
    timesteps, reward = np.loadtxt(fp, delimiter=',', unpack=True)
    ax.plot(timesteps, reward)
plt.show()
