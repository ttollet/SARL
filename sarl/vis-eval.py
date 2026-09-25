"""
Usage:
uv run python3 vis-eval.py /path/to/eval.csv
"""

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser(prog='vis-eval', description='Visualise eval.csv files.')
parser.add_argument('filepath')

args = parser.parse_args()
timesteps, reward = np.loadtxt(args.filepath, delimiter=',', unpack=True)
plt.plot(timesteps, reward)
plt.show()
