# %%
from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
import ray
import importlib as imp
import mdp_util
imp.reload(mdp_util)
from mdp_util import hodgkin_huxley_model, get_firing_rate
import numpy as np


# %%
hh_model = hodgkin_huxley_model()

parameters = {}
parameters['amp1'] = 150
parameters['amp2'] = 150
parameters['freq1'] = 2e2
parameters['freq2'] = 2.05e2
parameters['total_time'] = 500
parameters['plot_wf'] = False

results = hh_model.stimulate_neurons(parameters)

# %%

import matplotlib.pyplot as plt

x0 = results['amp_t']
y0 = results['amp_wf']

x1 = results['t_Pyr']
y1 = results['response_Pyr']
f1 = get_firing_rate(y1, x1)

x2 = results['t_PV']
y2 = results['response_PV']
f2 = get_firing_rate(y2, x2)

plt.figure()
plt.plot(x0, y0, c='k')
plt.plot(x1, y1, c='royalblue', label=f'Pyr {f1}')
plt.plot(x2, y2, c='orangered', label=f'PV {f2}')
plt.legend()
plt.show()
