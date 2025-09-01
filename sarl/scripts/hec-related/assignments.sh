#!/usr/bin/env bash

# Arrays
discrete_algs=("ppo" "a2c" "dqn")
continuous_algs=("ppo" "a2c" "ddpg" "sac" "td3")
envs=("platform" "goal")

num_discrete_algs=${#discrete_algs[@]}
num_continuous_algs=${#continuous_algs[@]}
num_envs=${#envs[@]}

# Functions
get_discrete_alg_for_id() {
  local task_id=$1
  local current_idx=$((task_id - 1))
  local item_idx=$(((current_idx / (num_envs * num_continuous_algs)) % num_discrete_algs))
  echo "${discrete_algs[$item_idx]}"
}

get_continuous_alg_for_id() {
  local task_id=$1
  local current_idx=$((task_id - 1))
  local item_idx=$(((current_idx / num_envs) % num_continuous_algs))
  echo "${continuous_algs[$item_idx]}"
}

get_env_for_id() {
  local task_id=$1
  local current_idx=$((task_id - 1))
  local item_idx=$((current_idx % num_envs))
  echo "${envs[$item_idx]}"
}

# Get max_id from CLI
max_id=$1
if [[ -z "$max_id" ]]; then
    echo "Usage: $0 <max_id>"
    exit 1
fi

# Declare associative array for env-discrete-continuous combinations
declare -A ids_by_combo
for env in "${envs[@]}"; do
  for d_alg in "${discrete_algs[@]}"; do
    for c_alg in "${continuous_algs[@]}"; do
      ids_by_combo["$env:$d_alg:$c_alg"]=""
    done
  done
done

# Loop over IDs
for id in $(seq 1 "$max_id"); do
    env=$(get_env_for_id "$id")
    d_alg=$(get_discrete_alg_for_id "$id")
    c_alg=$(get_continuous_alg_for_id "$id")

    ids_by_combo["$env:$d_alg:$c_alg"]+="$id, "
done

# Print output grouped by environment, header uppercase
for env in "${envs[@]}"; do
    echo "${env^^}"
    for d_alg in "${discrete_algs[@]}"; do
        for c_alg in "${continuous_algs[@]}"; do
            list=${ids_by_combo["$env:$d_alg:$c_alg"]%, }
            [[ -n "$list" ]] && echo "$env-$d_alg-$c_alg: [$list]"
        done
    done
done

