# %% Imports & Configurations
import os
from gymnasium.spaces import discrete
from matplotlib.artist import get
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams["lines.linewidth"] = 2

# DATES = ["2025-09-15", "2025-08-30", "2025-08-31", "2025-07-06", "2025-07-07"]
DATES = ["2025-09-15", "2025-09-16", "2025-09-17", "2025-09-18", "2025-09-19"]
ENVIRONMENT = {  # NB: Selections, set to 1 to plot, 0 to exclude
    "platform": 1,
    "goal": 0
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
BASELINES = ["qpamdp", "pdqn"]
CYCLE_LOW=1#99
CYCLE_HIGH=999#129
WINDOW_SIZE = 5  # -1 to disable
CI_WINDOW_SIZE = 15
MAX_TIMESTEP_OVERRIDE = 1000000 # Default: None
MIN_TIMESTEP_OVERRIDE = None  # Default: None
CONFIDENCE_INTERVALS = True
PLOT_TOP=3  # Default: 0
PLOT_NAME = None  # Default: None



# %% Data Wrangling
# - df_trials is mean_return at each timestep
def get_experiment_name(config):
    experiment = (config['environment'] + "-" +
                (config['algorithm'].replace("-", "+")) + "-" +
                # config['algorithm'] + "-" +
                # (str(config["parameters"]["cycles"]) + "cycles-" if "-" in config['algorithm'] else "") +
                # str(config["parameters"]["max_steps"]) + "maxSteps-" +
                str(config["parameters"]["eval_episodes"]) + "evalEpisodes")
    return experiment

trial_dfs = []
missing_csv = {}
experiment_counters = {}
def update_exp_counters(experiment):
    if experiment not in experiment_counters:
        experiment_counters[experiment] = 1
    experiment_counters[experiment] += 1

for date in DATES:  # Populate dataframe
    for trial_dir in os.listdir(f"../outputs/{date}"):
        config = yaml.safe_load(open(f"../outputs/{date}/{trial_dir}/.hydra/config.yaml"))

        # Handle missing eval.csv files
        if not os.path.exists(f"../outputs/{date}/{trial_dir}/eval.csv"):
            experiment = get_experiment_name(config)
            if experiment not in missing_csv: missing_csv[experiment] = []
            missing_csv[experiment].append(f"../outputs/{date}/{trial_dir}")
            continue

        experiment = get_experiment_name(config)
        update_exp_counters(experiment)
        discrete_alg = (config["algorithm"].split("-")[0] if "-" in config["algorithm"] else config["algorithm"])
        continuous_alg = (config["algorithm"].split("-")[1] if "-" in config["algorithm"] else config["algorithm"])

        df_trial = pl.read_csv(f"../outputs/{date}/{trial_dir}/eval.csv",
            new_columns=["training_timesteps", "mean_return"])
        df_trial = df_trial.with_columns([
            pl.lit(experiment).alias("experiment"),
            pl.lit(config["parameters"]["seeds"][0]).alias("seed"),
            pl.lit(config["parameters"]["cycles"]).alias("cycles"),
            pl.lit(config["environment"]).alias("environment"),
            pl.lit(config["algorithm"]).alias("algorithm"),
            pl.lit(discrete_alg).alias("discrete_alg"),
            pl.lit(continuous_alg).alias("continuous_alg"),
            pl.lit(range(1,df_trial.height+1)).alias("experiment_counter")])
        trial_dfs.append(df_trial)
df_trials = pl.concat(trial_dfs, rechunk=True)
df_trials.head()

# Data Wrangling Debug 0
# experiment_counters

# %% Data Wrangling Debug
# Count unique seeds per discrete-continuous algorithm pair
seed_counts = df_trials.filter(
    (pl.col("cycles") > CYCLE_LOW) & (pl.col("cycles") < CYCLE_HIGH)
).group_by(["discrete_alg", "continuous_alg", "environment"]).agg(
    pl.col("seed").n_unique().alias("n_unique_seeds")
).sort(["environment", "discrete_alg", "continuous_alg"])

print("Unique seeds per discrete-continuous algorithm pair:")
# print(seed_counts.filter(pl.col("discrete_alg") == "dqn"))
print(seed_counts["n_unique_seeds"].unique())

# %% Data Wrangling Debug 2
# Find number of trials per algorithm-environment pair
#num_trials_per_pair = df_trials.group_by(["algorithm", "environment"]).agg(pl.len().alias("num_trials"))
# num_trials_per_pair["num_trials"].min()
df_seeds_per_exp = df_trials.filter(
    (pl.col("cycles") > CYCLE_LOW) & (pl.col("cycles") < CYCLE_HIGH)
    ).group_by(["experiment", "cycles", "environment", "discrete_alg"]
    ).agg(pl.col("seed").n_unique().alias("n_seeds"))
for env in ["platform", "goal"]:
    for alg in BASELINES+["ppo", "dqn", "a2c"]:
        print(df_seeds_per_exp.filter((pl.col("environment") == env) & (pl.col("discrete_alg") == alg)))

# Missing experiments?
# print(missing_csv[list(missing_csv.keys())[0]])

# %% Pre-Plotting
# handle selections
env_selection = [env for env, selected in ENVIRONMENT.items() if selected]
alg_selection = [alg for alg, selected in DISCRETE_ALGS.items() if selected]
continuous_alg_selection = [alg for alg, selected in CONTINUOUS_ALGS.items() if selected]+BASELINES

# name plot
assert len(env_selection) == 1
if PLOT_NAME is None:
    plot_name = f"{env_selection[0].capitalize()}-V0 Evaluation Returns (n=50)"
else:
    plot_name = PLOT_NAME
# plot_file_name = plot_name.replace(" ", "_").lower()
plot_file_name = env_selection[0]

# construct df
df_plot = (df_trials
           .filter(pl.col("environment").is_in(env_selection))
           .filter(pl.col("discrete_alg").is_in(alg_selection))
           .filter(pl.col("continuous_alg").is_in(continuous_alg_selection))
           # .filter((pl.col("cycles") > CYCLE_LOW) & (pl.col("cycles") < CYCLE_HIGH))
)
df_plot.head()


# %% Aligning samples per baseline
# This is the problem, notice how each timestep is the first from that respective seed, but they are all different
# df_plot.filter(pl.col("algorithm").is_in(BASELINES), pl.col("training_timesteps") < 4000).head()
# for baseline in BASELINES:
baseline = BASELINES[0]  # Temp TODO



# %% Plotting

# compute statistics
# agg_df_init0 = df_plot.group_by(["algorithm", "training_timesteps"]).agg([  # Construct necessary dataframe
agg_df_init = df_plot.group_by(["algorithm", "experiment_counter"]).agg([  # Construct necessary dataframe
        pl.mean("training_timesteps"),
        pl.mean("mean_return").alias("ret_mean"),
        pl.std("mean_return").alias("ret_std"),
        (pl.count("mean_return") ** 0.5).alias("sqrt_n")
    ]).with_columns([  # 95% confidence interval = mean ± 1.96 * std/sqrt(n)
        (pl.col("ret_mean") - 1.96 * pl.col("ret_std")/pl.col("sqrt_n")).alias("ci_low"),
        (pl.col("ret_mean") + 1.96 * pl.col("ret_std")/pl.col("sqrt_n")).alias("ci_high"),
    ]).sort(["algorithm", "experiment_counter"
    ]).with_columns([
        (pl.col("ret_mean").rolling_mean(window_size=WINDOW_SIZE, center=True).alias("ret_mean_smooth")),
        (pl.col("ci_low").rolling_mean(window_size=CI_WINDOW_SIZE, center=True).alias("ci_low_smooth")),
        (pl.col("ci_high").rolling_mean(window_size=CI_WINDOW_SIZE, center=True).alias("ci_high_smooth"))
    ])

# %% Plotting
# Ensure identical spacing between samples of Baselines and Converter
# - Identify density of converter samples (rows per _ timesteps)
# - Enforce density on baselines (delete at uniform spacing to enforce)
# - Use gather_every to construct example replacement baseline data
# - Separate then merge
#
# converter_training_step = agg_df_init0.filter(pl.col("algorithm")=="ppo-ppo")["training_timesteps"][0]
# agg_df_just_baselines = agg_df_init0.filter(pl.col("algorithm").is_in([]))
# for baseline in BASELINES:
#     baseline_row_step = agg_df_init0.filter(pl.col("algorithm")==baseline, pl.col("training_timesteps")<converter_training_step).count()["algorithm"][0]
#     agg_df_just_baselines.extend(agg_df_init0.filter(pl.col("algorithm")==baseline).gather_every(baseline_row_step))
# agg_df_init = agg_df_init0.filter(~pl.col("algorithm").is_in(BASELINES)).vstack(agg_df_just_baselines)

# %% Debug cont.
agg_df_init.filter(pl.col("algorithm")=="qpamdp")

# %% Plotting cont.

# filter for best algorithms
if PLOT_TOP != 0:
    top_three = agg_df_init.group_by("algorithm").agg(
        pl.max("ret_mean_smooth").alias("max_ret_mean_smooth")
    ).sort("max_ret_mean_smooth", descending=True).filter(
        ~pl.col("algorithm").is_in(BASELINES)
    ).head(3)["algorithm"]
    agg_df = agg_df_init.filter(pl.col("algorithm").is_in(top_three) | pl.col("algorithm").is_in(BASELINES))
else:
    agg_df = agg_df_init

# enforce max timesteps
if MAX_TIMESTEP_OVERRIDE:
    agg_df = agg_df.filter(pl.col("training_timesteps") <= MAX_TIMESTEP_OVERRIDE)

# find the shared maximum timesteps across all algorithms
# df_plot.group_by(["algorithm", "seed"]).agg(pl.count("seed"))
# %% DEBUG smoothing since baselines are not smoothing as much, likely too frequent sample rate
agg_df.filter(pl.col("algorithm") == "ppo-ppo").head()
agg_df["algorithm"].unique()


for alg in agg_df["algorithm"].unique():
    sub_df = agg_df.filter(pl.col("algorithm") == alg)

    x = sub_df["training_timesteps"]
    if WINDOW_SIZE != -1:
        # Apply moving average smoothing
        y = sub_df["ret_mean_smooth"]
        window_size = min(WINDOW_SIZE, len(y))
        ci_low = sub_df["ci_low_smooth"]
        ci_high = sub_df["ci_high_smooth"]
    else:
        y = sub_df["ret_mean"]
        ci_low = sub_df["ci_low"]
        ci_high = sub_df["ci_high"]

    # Define line styles for discrete algorithms
    line_styles = {
        'pdqn': ':',
        'qpamdp': ':',
    }

    # Get the discrete algorithm for this algorithm
    discrete_alg = alg.split('+')[0] if '+' in alg else alg
    line_style = line_styles.get(discrete_alg, '-')  # default to solid if not found

    # Plot this algorithm
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
if ENVIRONMENT["platform"]:
    plt.ylim(0,1)
plt.xlim(0,MAX_TIMESTEP_OVERRIDE)
plt.savefig(filename)
# Display plot here
# plt.show()


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
def get_ret_from_alg(alg):
    alg_df = agg_df_init.filter(
        pl.col("algorithm") == alg
    )
    best_timestep = alg_df.filter(
            pl.col("ret_mean_smooth") == alg_df["ret_mean_smooth"].max()
        )["training_timesteps"].unique()
    alg_df_plot = df_plot.filter(
        pl.col("algorithm") == alg
    ).filter(
        (pl.col("training_timesteps") == best_timestep)
    )
    return alg_df_plot.sort("training_timesteps")["mean_return"].to_list()

for alg in algorithms:
    alg_data = get_ret_from_alg(alg)
    boxplot_data.append(alg_data)
    labels.append(alg.upper())

# Create boxplot
bp = ax.boxplot(boxplot_data, tick_labels=labels, patch_artist=False, vert=True, widths=0.7)

# Customize the plot
ax.set_xlabel("Algorithm")
ax.set_ylabel("Mean Evaluation Return")
plot_name = f"{env_selection[0].capitalize()}-V0 Best Policies (n=50)"
ax.set_title(plot_name)
if ENVIRONMENT["platform"]:
    plt.ylim(0,1)
plt.xticks(rotation=45)
plt.tight_layout()

# Export to png
filename = get_unique_filename("aggregate_outputs", str("BOX_"+plot_file_name+".png"))
plt.savefig(filename)

# Display here
# plt.show()
# %% Combination performance
discrete_algs = df_plot["discrete_alg"].unique().sort()
continuous_algs = df_plot["continuous_alg"].unique().sort()
df_mat_data = {"discrete_alg":[], "continuous_alg":[], "mean_return":[], "std_return":[]}
for alg1 in discrete_algs:
    for alg2 in continuous_algs:
        df_mat_data["discrete_alg"].append(alg1)
        df_mat_data["continuous_alg"].append(alg2)
        if alg1 in BASELINES:
            alg_data = get_ret_from_alg(alg1)
        else:
            alg_data = get_ret_from_alg(alg1+"-"+alg2)
        df_mat_data["mean_return"].append(np.mean(alg_data))
        df_mat_data["std_return"].append(np.std(alg_data))
df_mat = pl.DataFrame(df_mat_data)
df_mat_conv = df_mat.filter(
    ~pl.col("discrete_alg").is_in(BASELINES),
    ~pl.col("continuous_alg").is_in(BASELINES),
)
df_mat_base = df_mat.filter(
    pl.col("discrete_alg").is_in(BASELINES)
).drop("continuous_alg").unique()
df_mat_conv_best = df_mat_conv.sort(
    pl.col("mean_return"), descending=True
).head(3)

# %% Pivot table
df = df_mat_conv
# step 1: build formatted string
df_fmt = df.with_columns(
    # (pl.col("mean_return").round(2).cast(str) + " ± " + pl.col("std_return").round(2).cast(str)).alias("val")
    pl.concat_str([
        pl.col("mean_return").round(2).cast(pl.Utf8),
        pl.lit(" ± "),
        pl.col("std_return").round(2).cast(pl.Utf8)
    ]).alias("val")
)
df_fmt2 = df_mat_base.with_columns(
    # (pl.col("mean_return").round(2).cast(str) + " ± " + pl.col("std_return").round(2).cast(str)).alias("val")
    pl.concat_str([
        pl.col("mean_return").round(2).cast(pl.Utf8),
        pl.lit(" ± "),
        pl.col("std_return").round(2).cast(pl.Utf8)
    ])
)
# step 2: pivot
grid = df_fmt.pivot(
    values="val",
    index="continuous_alg",    # rows
    on="discrete_alg",  # columns
)


# Export to latex
grid_pd = grid.to_pandas().set_index("continuous_alg")
grid2_pd = df_fmt2.rename(
    {"discrete_alg": "Algorithm", "mean_return": "Mean Return & Std."}
).drop("std_return"
).to_pandas().set_index("Algorithm")

for grid in [grid_pd, grid2_pd]:
    grid.index = [str(i).upper() for i in grid.index]
grid_pd.columns = [col.upper() for col in grid_pd.columns]
grid_pd.index.name = "Continuous Alg."
grid_pd.columns.name = "Discrete Alg."
grid_pd = grid_pd.transpose()

filename = grid_pd.to_latex(get_unique_filename("aggregate_outputs", str("MATRIX_"+plot_file_name+".tex")), index=False)
grid2_pd.to_latex(get_unique_filename("aggregate_outputs", str("MATRIX_"+plot_file_name+"_base.tex")), index=False)

print(grid_pd)
