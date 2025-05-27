# Structured Action Reinforcement Learning (SARL)

> ⚠️ SARL is pre-release software, in active development!

SARL is a toolkit for performing reinforcement learning on environments with parameterised action spaces. Such action-spaces differ from the non-hierarchical, exclusively discrete or continuous action-spaces explored in conventional reinforcement learning.

A parameterized action-space, requires an agent to select from a discrete set of actions, then specify a continuous vector from that action's corresponding parameter-space [(Masson et. al. 2016)](https://doi.org/10.1609/aaai.v30i1.10226).

## Contents

### Baselines
- P-DQN

<!-- | Name     | Inclusion   | Notes
| -------- | ----------- | -
| P-DQN    |  ✅          | -->

<!-- | MP-DQN   |             |
| SP-DQN   |             |
| Q-PAMDP  |             |
| H-PPO    |             |
| PASVG(0) |             |
| PATRPO   |             |
| PADDPG   |             | -->

### Environments

- Platform
- Goal

<!-- | Name                     | Inclusion| Notes
| ------------------------ | -------- | -
| Platform | ✅ |
| Goal | ✅ | -->

<!-- | Chase & Attack           |          |
| Catching Point           |          |
| King of Glory            |          |
| Simulation / Moving      |          |-->

<!-- \* Custom wrappers exist, yet to be included in environment options. -->

## Usage

Instructions for use on Linux, Mac OSX, or Windows (via WSL).
* Ensure Python is installed `python --version`
* Ensure Poetry is installed `poetry --version`
    * You can install poetry with `curl -sSL https://install.python-poetry.org | python3 -`
    * Full instructions and alternative methods can be found [here](https://python-poetry.org/docs/#installing-with-the-official-installer)
* Clone this repository `git clone https://github.com/ttollet/SARL.git`
* Change directory `cd SARL`
* Install dependencies to virtual environment `poetry install`
* Check functionality `poetry run pytest`
* Enter virtual environment with `poetry shell`
    * Exit the virtual environment with `exit`

## Examples
```
python3 sarl/train.py algorithm=ppo-ppo environment=platform hydra.job.name=ppo-ppo-platform parameters.learning_steps=540000 parameters.seeds=[1] parameters.cycles=600
```