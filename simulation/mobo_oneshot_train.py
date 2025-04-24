# %%
from environment import HodgkinHuxley_Environment
import torch
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import matplotlib.pyplot as plt

train_cell_ids = [8, 10, 7]
test_cell_ids = [6, 9]

parameters = [
    {"name": "freq1", "type": "range", "value_type": "float", "bounds": [100, 2000]},
    {"name": "freq2", "type": "range", "value_type": "float", "bounds": [100, 2000]},
    {"name": "amp1", "type": "range", "value_type": "float", "bounds": [1e2, 20e3]},
    {"name": "amp2", "type": "range", "value_type": "float", "bounds": [1e2, 20e3]},
]

objectives = {
    "fr_diff": ObjectiveProperties(minimize=False, threshold=90),
    "nrg": ObjectiveProperties(minimize=True, threshold=0.0004)
}

ax_client = AxClient()
ax_client.create_experiment(
    name="",
    parameters=parameters,
    objectives=objectives
)

def evaluate(parameters, env):
    state, _ = env.reset()
    state, reward, terminated, truncated, _ = env.step([parameters['freq1'], parameters['freq2'], parameters['amp1'], parameters['amp2']])
    return {"fr_diff": (int(-state[0]), 0.0), "nrg": (state[1], 0.0)} # Return (mean, SEM)

for cell_id in test_cell_ids:
    env = HodgkinHuxley_Environment(algo="mobo")
    env.set_cell_id(cell_id)
    for i in range(1160):
        parameters, trial_index = ax_client.get_next_trial()
        evaluation = evaluate(parameters, env)
        ax_client.complete_trial(trial_index, raw_data=evaluation)
        env.storage.save(algo='mobo', postfix='test-1060')


# for cell_id in test_cell_ids:
#     env = HodgkinHuxley_Environment(algo="mobo")
#     env.set_cell_id(cell_id)
#     for i in range(30): # episode length
#         parameters, trial_index = ax_client.get_next_trial()
#         ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, env))
#     env.storage.save(algo="mobo", postfix="play")

# %%
# --- Analysis and Visualization ---
from ax.utils.notebook.plotting import render
objectives = ax_client.experiment.optimization_config.objective.objectives
frontier = compute_posterior_pareto_frontier(
    experiment=ax_client.experiment,
    data = ax_client.experiment.fetch_data(),
    primary_objective=objectives[0].metric,
    secondary_objective=objectives[1].metric,
    absolute_metrics=["fr_diff", "nrg"],
    num_points=40
)
render(plot_pareto_frontier(frontier, CI_level=0.90))
