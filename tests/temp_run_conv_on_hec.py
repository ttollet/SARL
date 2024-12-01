# -*- coding: utf-8 -*-
import sys
from tests.test_converter import test_converter_both

# learning_steps=250*2*3 is 53s on M1 MBP
CYCLES = 10
TARGET_MINUTES = 60 * 10

# Allow seed to be passed via command line
seed = 42
args = sys.argv
if len(args) > 1:
        print(f"Sys.argv = {sys.argv}")
        seed:int = int(args[1])
        discreteAlg = args[2]
        continuousAlg = args[3]
print(f"Seed = {seed}")

timesteps  = 250*2*3 * TARGET_MINUTES * CYCLES

test_converter_both(discreteAlg=discreteAlg, continuousAlg=continuousAlg, learning_steps=timesteps, log_results=True, cycles=CYCLES, seed=seed)
#test_converter_both(discreteAlg="PPO", continuousAlg="PPO", learning_steps=timesteps, log_results=True, cycles=CYCLES, seed=seed)
#test_converter_both(discreteAlg="A2C", continuousAlg="PPO", learning_steps=timesteps, log_results=True, cycles=CYCLES, seed=seed)
