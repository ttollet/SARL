# Structured Action Reinforcement Learning (SARL)

> ⚠️ SARL is pre-release software, in active development!

SARL is a toolkit for performing reinforcement learning on unconventionally specified action spaces. Such action-spaces differ from the non-hierarchical, exclusively discrete or continuous action-spaces explored in conventional reinforcement learning.

## Structures

### Parameterized Actions

The most common structured action space is the parameterized action-space, requiring an agent to select from a discrete set of actions, then specify a continuous vector from that action's corresponding parameter-space [(Masson et. al. 2016)](https://doi.org/10.1609/aaai.v30i1.10226).

## Contents

### Algorithms

| Name     | Inclusion   | Notes
| -------- | ----------- | -
| H-PPO    |             |
| MP-DQN   |             |
| SP-DQN   |             |
| P-DQN    | In Progress |
| PASVG(0) |             |
| PATRPO   |             |
| PADDPG   |             |
| Q-PAMDP  | In Progress |

### Environments

| Name                     | Inclusion| Notes
| ------------------------ | -------- | -
| Chase & Attack           |          |
| Catching Point           |          |
| King of Glory            |          |
| Simulation / Moving      |          |
| Half Field Offense (HFO) |          |
| Platform                 | ✔        | *
| RoboCup                  | ✔        | *

\* Custom wrappers exist, yet to be included in environment options.