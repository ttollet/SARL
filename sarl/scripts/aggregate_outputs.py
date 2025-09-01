# %% Imports & Configurations
import os
from gymnasium.spaces import discrete
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import yaml
DATES = ["2025-08-30", "2025-08-31", "2025-07-06", "2025-07-07"]
ENVIRONMENT = {
    "platform": 0,
    "goal": 1
}
DISCRETE_ALGS = {
    # Converter
    "ppo": 1,
    "a2c": 1,
    "dqn": 1,
    # Baselines
    "qpamdp": 1,
    "pdqn": 1
}
CONTINUOUS_ALGS = dict(zip(
    ["ddpg", "ppo", "td3", "a2c", "sac"],
    [1,1,1,1,1]
))
WINDOW_SIZE = 30
MAX_TIMESTEP_OVERRIDE = 600000 # Default: None
MIN_TIMESTEP_OVERRIDE = None  # Default: None
CONFIDENCE_INTERVALS = False
PLOT_NAME = None  # Default: None

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
        continuous_alg = (config["algorithm"].split("-")[1] if "-" in config["algorithm"] else config["algorithm"])
        df_trial = pl.read_csv(f"../outputs/{date}/{trial_dir}/eval.csv",
            new_columns=["training_timesteps", "mean_return"])
        df_trial = df_trial.with_columns([
            pl.lit(experiment).alias("experiment"),
            pl.lit(config["parameters"]["seeds"][0]).alias("seed"),
            pl.lit(config["environment"]).alias("environment"),
            pl.lit(config["algorithm"]).alias("algorithm"),
            pl.lit(discrete_alg).alias("discrete_alg"),
            pl.lit(continuous_alg).alias("continuous_alg")])
        trial_dfs.append(df_trial)
df_trials = pl.concat(trial_dfs, rechunk=True)
df_trials.columns


# %% Pre-Plotting
env_selection = [env for env, selected in ENVIRONMENT.items() if selected]
alg_selection = [alg for alg, selected in DISCRETE_ALGS.items() if selected]
continuous_alg_selection = [alg for alg, selected in CONTINUOUS_ALGS.items() if selected]+["pdqn", "qpamdp"]

assert len(env_selection) == 1
if PLOT_NAME is None:
    plot_name = f"{env_selection[0].capitalize()} with Baselines"
else:
    plot_name = PLOT_NAME
plot_file_name = plot_name.replace(" ", "_").lower()

df_plot = (df_trials
           .filter(pl.col("environment").is_in(env_selection))
           .filter(pl.col("discrete_alg").is_in(alg_selection))
           .filter(pl.col("continuous_alg").is_in(continuous_alg_selection)))
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
min_timesteps_per_alg = agg_df.group_by("algorithm").agg(pl.min("training_timesteps").alias("min_timesteps"))
if MAX_TIMESTEP_OVERRIDE is not None:
    shared_max_timesteps = MAX_TIMESTEP_OVERRIDE
else:
    shared_max_timesteps = max_timesteps_per_alg["max_timesteps"].min()
if MIN_TIMESTEP_OVERRIDE is not None:
    shared_min_timesteps = MIN_TIMESTEP_OVERRIDE
else:
    shared_min_timesteps = min_timesteps_per_alg["min_timesteps"].max()
agg_df = (agg_df
        .filter(pl.col("training_timesteps") <= shared_max_timesteps)
        .filter(pl.col("training_timesteps") >= shared_min_timesteps))


def moving_average(data, window_size=5):
    """Apply moving average smoothing to data"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

for alg in agg_df["algorithm"].unique():
    sub_df = agg_df.filter(pl.col("algorithm") == alg)

    x = sub_df["training_timesteps"]
    y = sub_df["ret_mean"]
    ci_low = sub_df["ci_low"]
    ci_high = sub_df["ci_high"]

    # Apply moving average smoothing
    window_size = min(WINDOW_SIZE, len(y))
    # Define line styles for discrete algorithms
    line_styles = {
        'pdqn': '--',     # dashed
        'qpamdp': ':',     # dash-dot
    }

    # Get the discrete algorithm for this algorithm
    discrete_alg = alg.split('+')[0] if '+' in alg else alg
    line_style = line_styles.get(discrete_alg, '-')  # default to solid if not found

    if len(y) >= window_size:
        y_smooth = moving_average(y.to_numpy(), window_size)
        ci_low_smooth = moving_average(ci_low.to_numpy(), window_size)
        ci_high_smooth = moving_average(ci_high.to_numpy(), window_size)
        x_smooth = x[window_size-1:]  # Adjust x to match smoothed data length

        plt.plot(x_smooth, y_smooth, label=alg, linestyle=line_style)
        if CONFIDENCE_INTERVALS:
            plt.fill_between(x_smooth, ci_low_smooth, ci_high_smooth, alpha=0.2)
    else:
        plt.plot(x, y, label=alg, linestyle=line_style)
        if CONFIDENCE_INTERVALS:
            plt.fill_between(x, ci_low, ci_high, alpha=0.2)

plt.xlabel("Training Timesteps")
plt.ylabel("Mean Evaluation Return")
plt.legend(title="Algorithm", bbox_to_anchor=(1.0, 1), loc="upper left")
plt.title(plot_name)
plt.tight_layout()
# Save plot with unique filename
most_recent = max(DATES)
def get_unique_filename(base_path, plot_file_name):
    counter = 1
    base_filename = f"{base_path}/{most_recent}_{plot_file_name}"
    filename = base_filename
    while os.path.exists(filename):
        name_parts = base_filename.rsplit('.', 1)
        filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
        counter += 1
    return filename

filename = get_unique_filename("aggregate_outputs", str("LINE_"+plot_file_name+".png"))
plt.savefig(filename)
# Display plot here
plt.show()


# %% Debug
# Count experiments per algorithm-environment combination
# experiment_counts = df_trials.group_by(["algorithm", "environment"]).agg(
#     pl.count().alias("num_experiments")
# ).sort(["environment", "algorithm"])

# print("Number of experiments per algorithm-environment combination:")
# print(experiment_counts)

# %% Boxplots
# Create boxplots of mean returns by algorithm
fig, ax = plt.subplots(figsize=(8, 6))

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
ax.set_ylabel("Mean Evaluation Return")
ax.set_title(plot_name)
plt.xticks(rotation=45)
plt.tight_layout()

# Export to png
filename = get_unique_filename("aggregate_outputs", str("BOX_"+plot_file_name+".png"))
plt.savefig(filename)

# Display here
plt.show()


# %% Combination performance
# Create performance matrix: discrete algs as columns, continuous algs as rows
discrete_algs = df_plot["discrete_alg"].unique().sort()
continuous_algs = df_plot["continuous_alg"].unique().sort()

# Calculate median and IQR for each combination
performance_data = []
for cont_alg in continuous_algs:
    row_data = []
    for disc_alg in discrete_algs:
        combo_data = df_plot.filter(
            (pl.col("discrete_alg") == disc_alg) &
            (pl.col("continuous_alg") == cont_alg)
        )
        if len(combo_data) > 0:
            median_val = combo_data["mean_return"].median()
            q25 = combo_data["mean_return"].quantile(0.25)
            q75 = combo_data["mean_return"].quantile(0.75)
            iqr_val = q75 - q25 if q25 is not None and q75 is not None else 0.0
            if iqr_val is not None:
                cell_value = f"{median_val:.2f}±{iqr_val:.2f}"
            else:
                cell_value = f"{median_val:.2f}±0.00"
        else:
            cell_value = "N/A"
        row_data.append(cell_value)
    performance_data.append(row_data)
# Create the dataframe
performance_matrix = pl.DataFrame(
    data=dict(zip([alg.upper() for alg in discrete_algs], zip(*performance_data))),
    schema={alg.upper(): pl.Utf8 for alg in discrete_algs}
).with_columns(
    pl.Series("Continuous Algorithm", [alg.upper() for alg in continuous_algs])
).select(["Continuous Algorithm"] + [alg.upper() for alg in discrete_algs])

# Export to png
filename = get_unique_filename("aggregate_outputs", str("MATRIX_"+plot_file_name+".tex"))
performance_matrix.to_pandas().to_latex(filename, index=False)

# Display here
print("Performance Matrix (Discrete Algs as Columns, Continuous Algs as Rows):")
print(performance_matrix)
