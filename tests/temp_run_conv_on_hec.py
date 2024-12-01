# -*- coding: utf-8 -*-
import sys
from tests.test_converter import test_converter_both

# learning_steps=250*2*3 is 53s on M1 MBP
CYCLES = 10
TARGET_MINUTES = 60

# Allow seed to be passed via command line
if len(sys.argv) == 1:
	seed = 42
else:
	assert sys.argv == 2
	seed = sys.argv[1]

timesteps  = 250*2*3 * TARGET_MINUTES * CYCLES

test_converter_both(discreteAlg="PPO", continuousAlg="PPO", learning_steps=timesteps, log_results=True, cycles=CYCLES, seed=)
test_converter_both(discreteAlg="A2C", continuousAlg="PPO", learning_steps=timesteps, log_results=True, cycles=CYCLES)
