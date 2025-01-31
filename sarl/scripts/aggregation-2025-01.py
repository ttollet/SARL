# This file is intended for us with an interactive python kernel via ipykernel, akin to a Jupyter notebook.
# The author used the Zed text editor's REPL feature.
# The feature is documented here: https://zed.dev/docs/repl
# poetry run python -m ipykernel install --user

# %% Constants - Changes as desired
GRAPH_SCALE_FACTOR = 2  # Size of output image, minimum of 1
WORKING_DIR = "/Users/tm0/github/SARL/"
OUTPUT_DIR = "outputs/"
SAVED_OUTPUTS_DIR = "outputs-saved"
SAVE_NAME = "first-hec-output"


# %% Import & configure as required
import os
import glob
import polars as pl
import altair as alt
# import seaborn as sns
import datetime
import math

# sns.set_theme(style="darkgrid")
alt.renderers.enable("browser")
alt.data_transformers.enable("vegafusion")


# %% Setup directories
save_dir = f"{SAVED_OUTPUTS_DIR}/{SAVE_NAME}"
os.makedirs(save_dir, exist_ok=True)


# %% Check working directory
os.chdir(WORKING_DIR)
print(os.getcwd())


# %% Find files
def find_csv_files(root_dir):
  """
  Finds all CSV files within a given directory and its subdirectories, including those with hyphens in their names.

  Args:
    root_dir: The root directory to start the search from.

  Returns:
    A list of strings, where each string is the full path to a CSV file.

  Via LLM
  """
  pattern = os.path.join(root_dir, '**', '*.csv')
  return glob.glob(pattern, recursive=True)

df1 = pl.DataFrame({"path": find_csv_files(OUTPUT_DIR)})
df1[0]


# %% Categorise paths by experiment
df2 = df1.with_columns(df1["path"].str.split('/').list.get(-2).alias("experiment"))
df2[0]


# %% Extract training curves - LONG RUNTIME ~1m
t_start = datetime.datetime.now()
n_experiments = len(df1)

def unpack_csv(path: str):
    csv = pl.read_csv(path)
    reward = csv.get_column("rollout/ep_rew_mean").to_numpy()
    timestep = csv.get_column("custom/pamdp_timestep").to_numpy()
    # type = csv.select("custom/agent_type")  # Continuous/Discrete
    return (timestep, reward)

experiment_labels = []
rewards = []
timesteps = []
# types = []

for row_index in range(n_experiments):
# for row_index in range(300):
    # csv_timesteps, csv_rewards, csv_types = unpack_csv(df2["path"][row_index])
    csv_timesteps, csv_rewards = unpack_csv(df2["path"][row_index])
    csv_length = len(csv_timesteps)
    samples = int(csv_length / 100)
    # samples = csv_length  # DO NOT ATTEMPT
    sample_interval = math.floor(csv_length / samples)
    for i in range(samples):
        sample_index = int(sample_interval * i)
        if sample_index >= csv_length:
            break
        try:
            timesteps.append(int(csv_timesteps[sample_index]))
            rewards.append(csv_rewards[sample_index])
            # types.append(csv_types[sample_index])
            experiment_labels.append(df2["experiment"][row_index])
        except:
            break

experiment_labels = pl.Series(experiment_labels)
rewards = pl.Series(rewards)
timesteps = pl.Series(timesteps)
# types = pl.Series(types)

t_end = datetime.datetime.now()
print(f"Runtime: {t_end - t_start}")

# %% Define final dataframe
df_final = pl.DataFrame({
    "timestep": timesteps,
    "reward": rewards,
    # "type": types,
    "sub_experiment": experiment_labels
})
df_final = df_final.with_columns(pl.col("sub_experiment").str.replace("discrete-", "").str.replace("continuous-", "").alias("experiment"))
df_final = df_final.with_columns(
    pl.when(pl.col("sub_experiment").str.contains("discrete"))
    .then(pl.lit("discrete"))
    .when(pl.col("sub_experiment").str.contains("continuous"))
    .then(pl.lit("continuous"))
    .otherwise(pl.lit("UNSPECIFIED"))
    .alias("agent_type")
)
df_final.write_csv(f"{save_dir}/results.csv")
df_final.head()
# %% Check the final dataframe
experiments = df_final.filter(pl.col("timestep")==0).group_by("experiment").agg(pl.len())
print(experiments)

# %% Select a single experiment
# df = df_final.filter(pl.col("experiment") == df2["experiment"][df_discrete_experiments["experiment"][0]])
# df.shape


# %% Plot results
def plot(df, title:str = "", n_samples = None, save:bool = True, open_in_browser:bool = False):
    experiment = df["experiment"][0]

    # frame = 100000
    line = alt.Chart(df).mark_line().encode(
        x="timestep",
        y="mean(reward)",
        color="agent_type"
    )
    # ).transform_window(
    #     rolling_mean='mean(reward)',
    #     frame=[-frame, frame]
    # )
    band = alt.Chart(
        df,
        title=alt.Title(
            title,
            subtitle=f"n={n_samples}"
        )
    ).mark_errorband(extent='ci').encode(
        x=alt.X("timestep").title("Timestep"),
        y=alt.Y("reward").title("Reward"),
        # color="agent_type"
    ).properties(
        width=1000,  # Set the width of the chart
        height=400  # Set the height of the chart
    )
    graph = band + line
    # graph = line

    # base = alt.Chart(df).mark_circle(opacity=0.5).transform_fold(
    #     fold=['A', 'B', 'C'],
    #     as_=['category', 'y']
    # ).encode(
    #     alt.X('x:Q'),
    #     alt.Y('y:Q'),
    #     alt.Color('category:N')
    # )
    # graph = base + base.transform_loess('x', 'y', groupby=['category']).mark_line(size=4)

    # SVGs
    # os.makedirs(f"{save_dir}/svg", exist_ok=True)
    # print("Saving "+f"{save_dir}/svg/{experiment}.svg")
    # graph.save(f"{save_dir}/svg/{experiment}.svg", scale_factor=GRAPH_SCALE_FACTOR)

    # PNGs
    os.makedirs(f"{save_dir}/png", exist_ok=True)
    print("Saving "+f"{save_dir}/png/{experiment}.png")
    graph.save(f"{save_dir}/png/{experiment}.png", scale_factor=GRAPH_SCALE_FACTOR)

    if open_in_browser:
        return graph
    else:
        return True

for i in range(len(experiments)):
    # experiment_to_plot = df_discrete_experiments["experiment"][i]
    # df_to_plot = df_final.filter(pl.col("experiment") == experiment_to_plot)
    # n_samples=df_discrete_experiments['len'][i]

    experiment_to_plot = experiments["experiment"][i]
    df_to_plot = df_final.filter(pl.col("experiment") == experiment_to_plot)
    n_samples = experiments["len"][i]

    plot(df_to_plot, title=experiment_to_plot, n_samples=n_samples, open_in_browser=True)
