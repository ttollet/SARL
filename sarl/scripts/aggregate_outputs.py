# %% Imports & Configurations
import os
import polars as pl
import altair as alt
DATES = ["2025-07-18"]#, "2025-07-07"]
EXCLUDED_EXPERIMENTS = [] #["pdqn-platform", "pdqn-goal", "--"]

# %% Group directories by experiment
dirs_by_experiment = {}
for date in DATES:
    path = f"../outputs/{date}"
    trial_dirs = os.listdir(path)
    experiments = set([f[9:] for f in trial_dirs])
    experiments = experiments - set(EXCLUDED_EXPERIMENTS)
    # dirs_by_experiment = {exp: [] for exp in experiments}
    # ^ TODO: FIX THIS LINE, IT OVERWRITES THE DIRS_BY_EXP DIR EACH TIME
    for exp in experiments:
        if exp not in dirs_by_experiment.keys():
            dirs_by_experiment[exp] = []
    for dir in trial_dirs:
        experiment = dir[9:]
        if experiment not in EXCLUDED_EXPERIMENTS:
            dirs_by_experiment[experiment].append(f"{date}/{dir}")
for experiment, dirs in dirs_by_experiment.items():
    print(f"{experiment}: {dirs[0]}, ...")


# %% Form dataframes
dfs_by_experiment = {}
not_found_csv_paths = []
not_found_experiment = []
for experiment, dirs in dirs_by_experiment.items():
    dfs = []
    for dir in dirs:
        csv_path = f"../outputs/{dir}/eval.csv"
        if os.path.exists(csv_path):
            df = pl.read_csv(csv_path)
            df.columns = ["training_timesteps", "mean_eval_episode_return"]
            df = df.with_columns(experiment = pl.lit(f"{experiment} (n={str(len(dirs))})"))
            dfs.append(df)
        else:
            not_found_csv_paths.append(csv_path)
    if not dfs:
        not_found_experiment.append(experiment)  # Ensure aggregation handles cases with numerous of same experiment
    else:
        df = pl.concat(dfs)
        dfs_by_experiment[experiment] = df
print("File(s) not found:")
print(*not_found_csv_paths, sep="\n")
print("\nNo dataframes found for experiments:")
print(*not_found_experiment, sep="\n")
print("\nDataframes:")
for experiment, df in dfs_by_experiment.items():
    print(f"{experiment}: {df.shape}")

# %% Plot
experiments = list(dfs_by_experiment.keys())
splits = [experiment.split("-") for experiment in experiments]
discrete_algs = set()
envs = set()
for split in splits:
    discrete_algs.add(split[0])
    envs.add(split[-1])

for discrete_alg in discrete_algs:
    for env in envs:
        plot_experiments = [e for e in experiments if e.startswith(discrete_alg) and e.endswith(env)]
        df = pl.concat([dfs_by_experiment[experiment] for experiment in plot_experiments])
        if not os.path.exists("aggregate_outputs"):
            os.makedirs("aggregate_outputs")

        plot = alt.Chart()
        base = alt.Chart(df, title=f"Discrete {discrete_alg.upper()} on {env.capitalize()}")
        line = base.mark_line().encode(
            x="training_timesteps",
            y="mean(mean_eval_episode_return)",
            color="experiment"
        )
        band = base.mark_area(opacity=0.2).encode(
            x=alt.X("training_timesteps").title("Training Timesteps"),
            y=alt.Y("ci0(mean_eval_episode_return)").title("Mean Return"),
            y2="ci1(mean_eval_episode_return)",
            color="experiment"
        )
        plot = band + line
        plot.save(f"aggregate_outputs/{discrete_alg}_X_{env}_plot.png")
