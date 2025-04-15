from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
import numpy as np
import ray
from scipy.signal import find_peaks

# intial set of params -> bounded
# agent chooses initial set
#
#

class hodgkin_huxley_model:
    """
    Environment to simulate HH neuron model.
    Initial params:
    """
    def __init__(self):
        pass

    def stimulate_neurons(self, parameters):
        amplitude1 = parameters['amp1']
        amplitude2 = parameters['amp2']
        frequency1 = parameters['freq1']
        frequency2 = parameters['freq2']
        total_time = parameters['total_time']
        plot_waveform = parameters['plot_wf']

        results = [
            simulation_Pyr.remote(
                    num_electrode=1,
                    amp1 = amplitude1, amp2 = amplitude2,
                    freq1 = frequency1, freq2 = frequency2,
                    total_time = total_time,
                    plot_waveform=plot_waveform # Set to True to plot injected current
            ),
            simulation_PV.remote(
                num_electrode=1,
                amp1=amplitude1, amp2 = amplitude2,
                freq1=frequency1, freq2=frequency2,
                total_time=total_time,
                plot_waveform=plot_waveform # Set to True to plot injected current
            )
        ]
        (response_Pyr, t_Pyr, amp_wf, amp_t), (response_PV, t_PV, amp_wf, amp_t) = ray.get(results)
        results = {
            'response_Pyr': response_Pyr,
            't_Pyr': t_Pyr,
            'response_PV': response_PV,
            't_PV': t_PV,
            'amp_wf': amp_wf,
            'amp_t': amp_t
        }
        return results


class markov_decision_process:
    """
    MDP + solver.
    Initial params:
        - environment: simulator
        - reward function: calculates rewards from states
        - discount_factor
    """
    def __init__(self, environment, reward_function, discount_factor):
        self.environment = environment
        self.reward_function = reward_function
        self.discount_factor = discount_factor

def get_firing_rate(membrane_potential, t):
    num_peaks, _ = find_peaks(membrane_potential.copy(), prominence=40)
    duration = np.max(t)/1000 # convert to seconds
    firing_rate = round(len(num_peaks)/duration) #firing rate
    return firing_rate

def calculate_features(simulation_results):

    # firing rates
    response_Pyr = simulation_results['response_pyr']
    t_Pyr = simulation_results['t_Pyr']
    response_PV = simulation_results['response_PV']
    t_PV = simulation_results['t_PV']

    fr_Pyr = get_firing_rate(response_Pyr, t_Pyr)
    fr_PV = get_firing_rate(response_PV, t_PV)
    fr_diff = fr_PV - fr_Pyr

    # energy efficiency
    amp_wf = simulation_results['amp_wf']
    amp_t = simulation_results['amp_t']
    nrg = np.sum(amp_wf ** 2) * (np.max(amp_t) / 1000)


    states = {
        'fr_diff': fr_diff,
        'energy': nrg,
    }

    return states




def reward_function(states, fr_coef=1.0, nrg_coef=1.0, fr_target=90, nrg_threshold=1000):

    pass
