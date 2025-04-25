# This file is intended for us with an interactive python kernel via ipykernel, akin to a Jupyter notebook.
# The author used the Zed text editor's REPL feature.
# The feature is documented here: https://zed.dev/docs/repl
# poetry run python -m ipykernel install --user

# %% Import as required
import os
import glob
import polars as pl
import matplotlib.pyplot as plt

# %% Setup directories
WORKING_DIR = "/Users/tm0/github/SARL/"
OUTPUT_DIR = "outputs/"
SAVED_OUTPUTS_DIR = "outputs-saved"
SAVE_NAME = "first-hec-output"
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
df_csvs = pl.DataFrame({
    "path": find_csv_files(OUTPUT_DIR)
})

# %%
# via llm
def split_path_column(df, column_name, new_column_names):
    """
    Splits a column containing file paths into multiple columns based on the '/' delimiter.

    Args:
        df: The Polars DataFrame.
        column_name: The name of the column containing the file paths.
        new_column_names: A list of strings representing the desired names for the new columns.

    Returns:
        A new DataFrame with the split columns.
    """

    # Split the path column into a list
    split_paths = df[column_name].str.split("/")

    # Create a dictionary to store the new columns
    new_columns = {}
    for i, name in enumerate(new_column_names):
        new_columns[name] = split_paths.list.get(i)

    # Add the new columns to the DataFrame
    return df.with_columns(**new_columns)
new_column_names = ["drop_me", "drop_me_2", "datetime", "seed", "details", "drop_me_3"]
df_csvs_aug = split_path_column(df_csvs, "path", new_column_names)
df_csvs_aug.drop_in_place("drop_me")
df_csvs_aug.drop_in_place("drop_me_2")
df_csvs_aug.drop_in_place("drop_me_3")

# Split date from time
df_csvs_aug = df_csvs_aug.with_columns([
    pl.col("datetime").str.split("_").list.get(0).alias("date"),
    pl.col("datetime").str.split("_").list.get(1).alias("time")
])
df_csvs_aug.drop_in_place("datetime")
df_csvs_aug.head()


# %% View count per experiment
grouped_details= df_csvs_aug.group_by("details").len().sort(by="len")
print(grouped_details)

# %% Form result CSV for each experiment
unique_experiments = df_csvs_aug["details"].unique()
experiment_dfs = {}

for experiment in unique_experiments:
    # Get paths for this experiment
    experiment_paths = df_csvs_aug.filter(pl.col("details") == experiment)["path"]

    # Read and concatenate all CSVs for this experiment
    dfs = []
    for path in experiment_paths:
        df = pl.read_csv(path)
        dfs.append(df)

    # Concatenate with matching columns
    experiment_dfs[experiment] = pl.concat(dfs, how="diagonal_relaxed")
# %% Check columns
experiment_dfs[unique_experiments[0]].columns
# Group experiments based on whether they're discrete or continuous
discrete_experiments = [exp for exp in unique_experiments if "discrete" in exp]
continuous_experiments = [exp for exp in unique_experiments if "continuous" in exp]

print("=== Column counts for each experiment pair ===")
for d, c in zip(discrete_experiments, continuous_experiments):
    print(f"\nDiscrete: {d}")
    print(f"Columns: {len(experiment_dfs[d].columns)}")
    print(f"\nContinuous: {c}")
    print(f"Columns: {len(experiment_dfs[c].columns)}")

# %% Graph results
# For each experiment, plot rollout/ep_rew_mean against custom/pamdp_timestep
for experiment in unique_experiments:
    df = experiment_dfs[experiment]
    plt.figure(figsize=(10, 6))
    plt.plot(df["custom/pamdp_timestep"], df["rollout/ep_rew_mean"])
    plt.title(f"Episode Reward Mean vs. Timestep - {experiment}")
    plt.xlabel("Timestep")
    plt.ylabel("Episode Reward Mean")
    plt.grid(True)
    plt.savefig(f"{save_dir}/{experiment}_reward.png")
    plt.close()
