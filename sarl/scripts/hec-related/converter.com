#!/bin/bash

#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -J trial-converter
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=9:00:00
# ^ Per array component

# Iterate over seeds
# n=x where x==algs*envs*trials
# alg*envs == (3*5)*2 = 30
# n=1 -> 1-30
# n=15 -> 1-450
# n=50 -> 1-1500
#SBATCH -a 1-1500

SEED_OFFSET=2000
TIMESTEPS=1001472  # Multiple of 2048
CYCLES=128  # 128 or other factor of Timesteps, like powers of 2
# default T=600000 at 100cycles needing 5:00:00

duration_start=$(date +%s)

source /etc/profile
module add cuda/12.0

echo This is job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}
echo Job running on compute node `uname -n`

# == SCRIPT STARTS HERE ==
id=${SLURM_ARRAY_TASK_ID}

# Array definitions
discrete_algs=("ppo" "a2c" "dqn")
continuous_algs=("ppo" "a2c" "ddpg" "sac" "td3")
envs=("platform" "goal")

# Get the number of items in each list
num_discrete_algs=${#discrete_algs[@]}
num_continuous_algs=${#continuous_algs[@]}
num_envs=${#envs[@]}

# Functions
get_discrete_alg_for_id() {
  local task_id=$1
  local current_idx=$((task_id - 1)) # Convert to 0-based index
  local item_idx=$(((current_idx / (num_envs * num_continuous_algs)) % num_discrete_algs))
  echo "${discrete_algs[$item_idx]}"
}

get_continuous_alg_for_id() {
  local task_id=$1
  local current_idx=$((task_id - 1)) # Convert to 0-based index
  local item_idx=$(((current_idx / num_envs) % num_continuous_algs))
  echo "${continuous_algs[$item_idx]}"
}

get_env_for_id() {
  local task_id=$1
  local current_idx=$((task_id - 1)) # Convert to 0-based index
  local item_idx=$((current_idx % num_envs))
  echo "${envs[$item_idx]}"
}

# Get the parts of the combination for the current id
first_alg=$(get_discrete_alg_for_id "$id")
second_alg=$(get_continuous_alg_for_id "$id")
env=$(get_env_for_id "$id")

# Form the final combination names
alg=$first_alg-$second_alg
seed=$((id+SEED_OFFSET))
name="${first_alg}-${second_alg}-${env}-seed${seed}"

echo "--- Configuration for Task ID ${id} ---"
echo "Discrete Alg: ${first_alg}"
echo "Continuous Alg: ${second_alg}"
echo "Environment: ${env}"
echo "Name: ${name}"
echo "------------------------------------"

module add miniforge
echo ==DIR==
echo >cd /storage/users/tollet/SARL
cd /storage/users/tollet/SARL
echo >pwd
pwd
echo ==CONDA==
echo >source activate /storage/users/tollet/conda_envs/sarl_env
source activate /storage/users/tollet/conda_envs/sarl_env
echo ==PYTHON==
echo >which python3
which python3
echo ==POETRY==
echo >which poetry
which poetry
echo ==RUN==
echo >python3 train.py hydra.job.name=$name algorithm=$alg environment=$env
python3 /storage/users/tollet/SARL/sarl/train.py hydra.job.name=$name algorithm=$alg environment=$env parameters.seeds=[$seed] parameters.learning_steps=$TIMESTEPS parameters.train_episodes=$TIMESTEPS
echo Finished!

duration_end=$(date +%s)
duration=$((duration_end - duration_start))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))
echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s (=${duration}s)"
echo "Time left: $(squeue -h -j $SLURM_JOB_ID -O TimeLeft)"
echo ==END==
