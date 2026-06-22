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

### Option 1: Local Setup (Linux, Mac OSX, or Windows via WSL)

* Ensure Python is installed `python --version`
* Ensure uv is installed `uv --version`
    * You can install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh`
    * Full instructions and alternative methods can be found [here](https://docs.astral.sh/uv/getting-started/installation/)
* Clone this repository `git clone https://github.com/ttollet/SARL.git`
* Change directory `cd SARL`
* Install dependencies to virtual environment `uv sync`
* Check functionality `uv run pytest`
* Activate virtual environment with `source .venv/bin/activate`
    * Exit the virtual environment with `deactivate`

### Option 2: Using Dev Containers

* Install [Docker](https://docs.docker.com/get-docker/)
* Install [Visual Studio Code](https://code.visualstudio.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
* Clone this repository and open it in VS Code
* Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select "Dev Containers: Reopen in Container"
* The environment will be automatically set up and ready to use

## Examples
```
python3 sarl/train.py algorithm=ppo-ppo environment=platform hydra.job.name=ppo-ppo-platform parameters.learning_steps=540000 parameters.seeds=[1] parameters.cycles=600
```
