# %%
from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
import ray
import importlib as imp
import environment
imp.reload(environment)
from environment import HodgkinHuxley_Model
import numpy as np


# %%
hh_model = HodgkinHuxley_Model()

parameters = {}
parameters['amp1'] = 1662
parameters['amp2'] = 1767
parameters['freq1'] = 15200
parameters['freq2'] = 17700
parameters['total_time'] = 500
parameters['plot_wf'] = False
parameters['stim_type'] = 'temporal_interference'

results = hh_model.stimulate_neurons(parameters)

# %%

import matplotlib.pyplot as plt

x0 = results['s1_input_time']
y0 = results['s1_input']

x1 = results['s2_input_time']
y1 = results['s2_input']
#f1 = get_firing_rate(y1, x1)

x2 = results['d_input_time']
y2 = results['d_input']
#f2 = get_firing_rate(y2, x2)

limit = 2000

plt.figure()
plt.plot(x0[0:limit], y0[0:limit], c='royalblue', label='shallow 1')
plt.plot(x1[0:limit], y1[0:limit], c='orangered', label='shallow 2')
plt.plot(x2[0:limit], y2[0:limit], c='green', label=f'deep')
plt.legend()
plt.show()


# %%
t = np.arange(0, 1, 0.00001)

a1, a2 = 1, 1
f1, f2, = 100, 101

s1 = a1 * np.sin(2*np.pi*f1*t)
s2 = a2 * np.sin(2*np.pi*f2*t)
#s3 = (a1+a2) * np.cos(np.pi*(f2-f1)*t) * np.sin(np.pi*(f1+f2)*t)
s3 = s1 + s2

plt.figure()
plt.plot(t, s1,)
plt.plot(t, s2,)
plt.plot(t, s3,)
plt.show()


t


# %%
np.sum(amp_wf ** 2) * (np.max(amp_t) / 1000)
