#!/bin/bash

#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -J sarl-baselines
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

# Iterate over seeds
# Use 1-x where x=algs*envs*n
#SBATCH -a 1-200

source /etc/profile
module add cuda/12.0

echo This is job task ${SLURM_ARRAY_TASK_ID}
echo Job running on compute node `uname -n`

# == SCRIPT STARTS HERE ==
id=${SLURM_ARRAY_TASK_ID}

# Array definitions
algs=("pdqn" "qpamdp")
envs=("platform" "goal")

# Get the number of items in each list
num_algs=${#algs[@]}
num_envs=${#envs[@]}

# Functions
get_alg_for_id() {
  local task_id=$1
  local current_idx=$((task_id - 1)) # Convert to 0-based index
  local item_idx=$(((current_idx / (num_envs)) % num_algs))
  echo "${algs[$item_idx]}"
}

get_env_for_id() {
  local task_id=$1
  local current_idx=$((task_id - 1)) # Convert to 0-based index
  local item_idx=$((current_idx % num_envs))
  echo "${envs[$item_idx]}"
}

# Get the parts of the combination for the current id
first_alg=$(get_alg_for_id "$id")
env=$(get_env_for_id "$id")

# Form the final combination names
alg=$first_alg
name="${first_alg}-${env}-${id}"

echo "--- Configuration for Task ID ${id} ---"
echo "Algorithm: ${first_alg}"
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
python3 /storage/users/tollet/SARL/sarl/train.py hydra.job.name=$name algorithm=$alg environment=$env parameters.seeds=[${SLURM_ARRAY_TASK_ID}] parameters.train_episodes=600000
echo Finished!
echo ==END==
