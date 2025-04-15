from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
import ray
from mdp_util import hodgkin_huxley_model, get_firing_rate
import numpy as np

hh_model = hodgkin_huxley_model()

parameters = {}
parameters['amp1'] = 150
parameters['amp2'] = 50
parameters['freq1'] = 30
parameters['freq2'] = 90
parameters['total_time'] = 500
parameters['plot_wf'] = True

results = hh_model.stimulate_neurons(parameters)
