# %% Imports & Configurations
import os
from gymnasium.spaces import discrete
import polars as pl
import matplotlib.pyplot as plt
import yaml
DATES = ["2025-08-30", "2025-08-31", "2025-07-06", "2025-07-07"]
# DATES = ["2025-07-06"]

# %% Data Wrangling
def get_experiment_name(config):
    experiment = (config['environment'] + "-" +
                config['algorithm'] + "-" +
                (str(config["parameters"]["cycles"]) + "cycles-" if "-" in config['algorithm'] else "") +
                str(config["parameters"]["max_steps"]) + "maxSteps-" +
                str(config["parameters"]["eval_episodes"]) + "evalEpisodes")
    return experiment

trial_dfs = []
missing_csv = {}
for date in DATES:  # Populate dataframe
    for trial_dir in os.listdir(f"../outputs/{date}"):
        if not os.path.exists(f"../outputs/{date}/{trial_dir}/eval.csv"):
            config = yaml.safe_load(open(f"../outputs/{date}/{trial_dir}/.hydra/config.yaml"))
            experiment = get_experiment_name(config)
            if experiment not in missing_csv: missing_csv[experiment] = []
            missing_csv[experiment].append(f"../outputs/{date}/{trial_dir}")
            continue
        config = yaml.safe_load(open(f"../outputs/{date}/{trial_dir}/.hydra/config.yaml"))
        experiment = (config['environment'] + "-" +
                     (config['algorithm'].replace("-", "+")) + "-" +
                     (str(config["parameters"]["cycles"]) + "cycles-" if "-" in config['algorithm'] else "") +
                     str(config["parameters"]["max_steps"]) + "maxSteps-" +
                     str(config["parameters"]["eval_episodes"]) + "evalEpisodes")
        discrete_alg = (config["algorithm"].split("-")[0] if "-" in config["algorithm"] else config["algorithm"])
        df_trial = pl.read_csv(f"../outputs/{date}/{trial_dir}/eval.csv",
            new_columns=["training_timesteps", "mean_return"])
        df_trial = df_trial.with_columns([
            pl.lit(experiment).alias("experiment"),
            pl.lit(config["parameters"]["seeds"][0]).alias("seed"),
            pl.lit(config["environment"]).alias("environment"),
            pl.lit(config["algorithm"]).alias("algorithm"),
            pl.lit(discrete_alg).alias("discrete_alg")])
        trial_dfs.append(df_trial)
df_trials = pl.concat(trial_dfs, rechunk=True)
df_trials.columns


# %% Pre-Plotting
ENVIRONMENT = {
    "platform": True,
    "goal": False
}
DISCRETE_ALGS = {
    # Converter
    "ppo": True,
    "a2c": False,
    "dqn": False,
    # Baselines
    "qpamdp": True,
    "pdqn": True
}
CONTINUOUS_ALGS = dict(zip(
    ["ddpg", "ppo", "td3", "a2c", "sac"],
    [1,1,1,1,1]
))
PLOT_FILE_NAME = "platform-best-v-baselines"

alg_selection = [alg for alg, selected in DISCRETE_ALGS.items() if selected]
env_selection = [env for env, selected in ENVIRONMENT.items() if selected]
assert len(env_selection) == 1
df_plot = (df_trials
           .filter(pl.col("discrete_alg").is_in(alg_selection))
           .filter(pl.col("environment").is_in(env_selection)))
df_plot.head()

# %% Plotting
agg_df = df_plot.group_by(["algorithm", "training_timesteps"]).agg([  # Construct necessary dataframe
        pl.mean("mean_return").alias("ret_mean"),
        pl.std("mean_return").alias("ret_std"),
        (pl.count("mean_return") ** 0.5).alias("sqrt_n")
    ]).with_columns([  # 95% confidence interval = mean ± 1.96 * std/sqrt(n)
        (pl.col("ret_mean") - 1.96 * pl.col("ret_std")/pl.col("sqrt_n")).alias("ci_low"),
        (pl.col("ret_mean") + 1.96 * pl.col("ret_std")/pl.col("sqrt_n")).alias("ci_high")
    ]).sort(["algorithm", "training_timesteps"])

# Find the shared maximum timesteps across all algorithms
max_timesteps_per_alg = agg_df.group_by("algorithm").agg(pl.max("training_timesteps").alias("max_timesteps"))
shared_max_timesteps = max_timesteps_per_alg["max_timesteps"].min()
agg_df = agg_df.filter(pl.col("training_timesteps") <= shared_max_timesteps)

for alg in agg_df["algorithm"].unique():
    sub_df = agg_df.filter(pl.col("algorithm") == alg)

    x = sub_df["training_timesteps"]
    y = sub_df["ret_mean"]
    ci_low = sub_df["ci_low"]
    ci_high = sub_df["ci_high"]

    plt.plot(x, y, label=alg)
    plt.fill_between(x, ci_low, ci_high, alpha=0.2)

plt.xlabel("Training Timesteps")
plt.ylabel("Mean Return")
plt.legend(title="Algorithm", bbox_to_anchor=(1.0, 1), loc="upper left")
plt.title(f"Evaluation Return by Algorithm ({env_selection[0].capitalize()})")
plt.tight_layout()
plt.savefig(f"aggregate_outputs/{PLOT_FILE_NAME}.png")
plt.show()

# %% Debug
# Count experiments per algorithm-environment combination
experiment_counts = df_trials.group_by(["algorithm", "environment"]).agg(
    pl.count().alias("num_experiments")
).sort(["environment", "algorithm"])

print("Number of experiments per algorithm-environment combination:")
print(experiment_counts)

# %% Boxplots
# Create boxplots of mean returns by algorithm
fig, ax = plt.subplots(figsize=(4, 6))

# Get data for boxplot
algorithms = agg_df["algorithm"].unique().sort()
boxplot_data = []
labels = []

for alg in algorithms:
    alg_data = agg_df.filter(pl.col("algorithm") == alg)["ret_mean"].to_list()
    boxplot_data.append(alg_data)
    labels.append(alg)

# Create boxplot
bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=False)

# Customize the plot
ax.set_xlabel("Algorithm")
ax.set_ylabel("Mean Return")
ax.set_title(f"Distribution of Mean Returns by Algorithm ({env_selection[0].capitalize()})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
