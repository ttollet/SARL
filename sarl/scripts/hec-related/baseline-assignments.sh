#!/bin/bash

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

# Get max_id from CLI
max_id=$1
if [[ -z "$max_id" ]]; then
    echo "Usage: $0 <max_id>"
    exit 1
fi

# Associative array to hold IDs per env+alg
declare -A ids_by_env_alg
for env in "${envs[@]}"; do
  for alg in "${algs[@]}"; do
    ids_by_env_alg["$env:$alg"]=""
  done
done

# Loop over IDs
for id in $(seq 1 "$max_id"); do
    env=$(get_env_for_id "$id")
    alg=$(get_alg_for_id "$id")
    ids_by_env_alg["$env:$alg"]+="$id, "
done

# Print output grouped by environment
for env in "${envs[@]}"; do
    # Header in all caps
    echo "${env^^}"
    for alg in "${algs[@]}"; do
        list=${ids_by_env_alg["$env:$alg"]%, }
        echo "$env-$alg: [$list]"
    done
done
