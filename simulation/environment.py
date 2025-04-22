from simulation_Pyr import simulation_Pyr
from simulation_PV import simulation_PV
import numpy as np
import ray
from scipy.signal import find_peaks
import gymnasium as gym


class HodgkinHuxley_Environment(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=np.array([1, 1, 1, 1]),
            high=np.array([2e3, 2e3, 5, 5]),
            shape=(4,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-150, 0]),
            high=np.array([150, 1e4]),
            shape=(),
            dtype=np.float32
        )
        self.model = HodgkinHuxley_Model()
        self.current_step = 0
    def reset(self, *, seed=None, options=None):
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        info = {}
        return self.state, info
    def step(self, action):
        """
        action: stimulus parameters
        """
        # simulation results
        results = self.model.stimulate_neurons(self._map_action(action), type='temporal_interferece')
        # calculate next state
        self.state = self.calc_state(results)
        # calculate reward & episode terminality
        reward, terminated = self.calc_reward(self.state)
        truncated = False
        self.current_step += 1
        info = {}
        return self.state, reward, terminated, truncated, info #, info
    def close(self):
        pass

    def get_firing_rate(self, membrane_potential, t):
        num_peaks, _ = find_peaks(membrane_potential.copy(), prominence=40)
        duration = np.max(t)/1000 # convert to seconds
        firing_rate = round(len(num_peaks)/duration) #firing rate
        return firing_rate

    def calc_state(self, simulation_results):
        # firing rates
        response_Pyr = simulation_results['response_pyr']
        t_Pyr = simulation_results['t_Pyr']
        response_PV = simulation_results['response_PV']
        t_PV = simulation_results['t_PV']
        fr_Pyr = self.get_firing_rate(response_Pyr, t_Pyr)
        fr_PV = self.get_firing_rate(response_PV, t_PV)
        fr_diff = fr_PV - fr_Pyr
        # energy efficiency
        amp_wf = simulation_results['amp_wf']
        amp_t = simulation_results['amp_t']
        nrg = np.sum(amp_wf ** 2) * (np.max(amp_t) / 1000)
        state = np.array([fr_diff, nrg], dtype=np.float32)
        return state
    def calc_reward(self, state, fr_coef=1.0, nrg_coef=1.0, fr_target=90, nrg_target=1000):
        fr_diff = state['fr_diff']
        nrg = state['energy'] # energy HAHA GET IT skibidi
        fr_reward = fr_coef * (fr_diff - fr_target)
        nrg_reward = nrg_coef * (nrg_target - nrg)
        sum_reward = fr_reward + nrg_reward
        return sum_reward
    def _map_action(self, action):
        return {
            'stim_type': 'PV-Pyr',
            'amp1': action[0],
            'amp2': action[1],
            'freq1': action[2],
            'freq2': action[3],
            'total_time': 500.0, # Example
            'plot_wf': False      # Example
        }


class HodgkinHuxley_Model:
    """
    Environment to simulate HH neuron model.
    Initial params:
    """
    def __init__(self):
        pass

    def stimulate_neurons(self, parameters, type='temporal_interference'):
        stimulation_type = parameters['stim_type']
        amplitude1, amplitude2 = parameters['amp1'], parameters['amp2']
        frequency1, frequency2 = parameters['freq1'], parameters['freq2']
        total_time = parameters['total_time']
        plot_waveform = parameters['plot_wf']

        if stimulation_type == 'temporal_interference':
            results = [
                simulation_Pyr.remote(
                    num_electrode=1,
                    amp1 = amplitude1, amp2 = 0,
                    freq1 = frequency1, freq2 = 0,
                    total_time = total_time,
                    plot_waveform=plot_waveform # Set to True to plot injected current
                ),
                simulation_Pyr.remote(
                    num_electrode=1,
                    amp1=0, amp2 = amplitude2,
                    freq1=0, freq2=frequency2,
                    total_time=total_time,
                    plot_waveform=plot_waveform # Set to True to plot injected current
                ),
                simulation_Pyr.remote(
                    num_electrode=1,
                    amp1=amplitude1, amp2 = amplitude2,
                    freq1=frequency1, freq2=frequency2,
                    total_time=total_time,
                    plot_waveform=plot_waveform # Set to True to plot injected current
                ),
            ]
            (o_s1, ot_s1, i_s1, it_s1), (o_s2, ot_s2, i_s2, it_s2), (o_s3, ot_s3, i_s3, it_s3) = ray.get(results)
            pass
        else: #stimulation_type == 'PV-Pyr':
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
