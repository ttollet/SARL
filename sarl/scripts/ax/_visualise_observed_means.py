# %% Imports
import os
from ax import Client
from ax.analysis.plotly.surface.slice import SlicePlot

# %% Load data
client_path = "./runs/bayesian/proper/2026-05-08_08-45/wip-client.json"  # <- More recent
#client_path = "./runs/bayesian/proper/incomplete/2026-05-08_08-33/wip-client.json"

client = Client.load_from_json_file(filepath=client_path)
analyses = client.compute_analyses(
    analyses=[
        SlicePlot(
            parameter_name="discrete_learning_rate",
            metric_name="mean_reward",
            display_sampled=True,
        ),
    ],
    display=False,
)

df = client.summarize()
x = df["discrete_learning_rate"]
y = df["mean_reward"]

# %% Visualise observed means
fig = analyses[0].get_figure()
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    marker=dict(
        symbol="circle",
        color="red",
        size=10,
        opacity=1.0,
        line=dict(color="black", width=1),
    ),
    name="Observed mean_reward",
    showlegend=True,
))
fig.update_xaxes(type="log")
fig.write_html(out, include_plotlyjs="cdn")
