# %%
from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
import ray
from environment import HodgkinHuxley_Environment, HodgkinHuxley_Model
import numpy as np

hh_model = HodgkinHuxley_Model()
parameters = {}
parameters['freq1'] = 1662
parameters['freq2'] = 1767
parameters['amp1'] = 15200
parameters['amp2'] = 17700
parameters['total_time'] = 500
parameters['plot_wf'] = False
parameters['stim_type'] = 'temporal_interference'

results = hh_model.stimulate_neurons(parameters, type='temporal_interference')
print(results)

# %%
import matplotlib.pyplot as plt
print(results)
plt.figure()
plt.title('Shallow Neuron 1')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Response (muA)')
plt.plot(results['s1_output_time'], results['s1_output'])
plt.figure()
plt.title('Shallow Neuron 2')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Response (muA)')
plt.plot(results['s2_output_time'], results['s2_output'], c='orangered')
plt.figure()
plt.title('Deep Neuron')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Response (muA)')
plt.plot(results['d_output_time'], results['d_output'], c='green')
plt.show()
print(np.max(results['s1_input']))
# %%
import environment
import importlib as imp
imp.reload(environment)
from environment import HodgkinHuxley_Environment, HodgkinHuxley_Model
import numpy as np

hh_env = HodgkinHuxley_Environment()
action = [1138.0, 852.0, 9.23, 20.12]
results2 = hh_env.step(action)

# %%
results2


# %%
plt.figure()
plt.plot()
plt.show()
