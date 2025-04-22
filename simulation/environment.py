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
            low=np.array([1, 1, 5, 5]),
            high=np.array([2e3, 2e3, 20e3, 20e3]),
            shape=(4,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-110, 0]),
            high=np.array([110, 0.0004]),
            shape=(2, ),
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
        results = self.model.stimulate_neurons(self._map_action(action), type='temporal_interference')
        # calculate next state
        self.state, terminated = self.calc_state(results)
        # calculate reward & episode terminality
        reward = self.calc_reward(self.state)
        truncated = False
        self.current_step += 1
        info = {}

        print('PINEAPPLE\n\n')
        print(action)
        print(self.state)
        print(reward)
        print(terminated)
        print(truncated)

        print(info)
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

        # response_Pyr = simulation_results['response_Pyr']
        # t_Pyr = simulation_results['t_Pyr']
        # response_PV = simulation_results['response_PV']
        # t_PV = simulation_results['t_PV']
        # fr_Pyr = self.get_firing_rate(response_Pyr, t_Pyr)
        # fr_PV = self.get_firing_rate(response_PV, t_PV)
        # fr_diff = fr_PV - fr_Pyr

        s1o_neuron, s1o_time = simulation_results['s1_output'], simulation_results['s1_output_time']
        s1i_stim, s1i_time = simulation_results['s1_input'], simulation_results['s1_input_time']
        s2o_neuron, s2o_time = simulation_results['s2_output'], simulation_results['s2_output_time']
        s2i_stim, s2i_time = simulation_results['s2_input'], simulation_results['s2_input_time']
        do_neuron, do_time = simulation_results['d_output'], simulation_results['d_output_time']
        fr_s1 = self.get_firing_rate(s1o_neuron, s1o_time)
        fr_s2 = self.get_firing_rate(s2o_neuron, s2o_time)
        fr_d = self.get_firing_rate(do_neuron, do_time)
        fr_diff = fr_d - ((fr_s1 + fr_s2) / 2)
        terminated = False
        if fr_diff >= 90:
            terminated = True
        # energy efficiency
        #amp_wf = simulation_results['amp_wf']
        #amp_t = simulation_results['amp_t']
        nrg_shallow1 = np.sum((s1i_stim / 1e6) ** 2) * (s1i_time[1] - s1i_time[0]) / 1000
        nrg_shallow2 = np.sum((s2i_stim / 1e6) ** 2) * (s2i_time[1] - s2i_time[0]) / 1000
        # nrg = np.sum(amp_wf ** 2) * (np.max(amp_t) / 1000)
        nrg = nrg_shallow1 + nrg_shallow2
        state = np.array([fr_diff, nrg], dtype=np.float32)
        return state, terminated
    def calc_reward(self, state, fr_coef=0.6, nrg_coef=0.4, fr_target=50, nrg_target=0.0004):
        nrg_target = self.observation_space.high[1] ** 2 * 500e-3 * 2 # calculate nrg target from upper bound
        fr_diff = state[0]
        nrg = state[1] # energy HAHA GET IT skibidi
        norm_fr_diff = self.normalize_firing_rate(fr_diff)
        norm_nrg = self.normalize_nrg(nrg)
        norm_fr_target = self.normalize_firing_rate(fr_target)
        norm_nrg_target = self.normalize_nrg(nrg_target)
        fr_reward = fr_coef * (norm_fr_diff - norm_fr_target)
        nrg_reward = nrg_coef * (norm_nrg_target - norm_nrg)
        sum_reward = fr_reward + nrg_reward
        return sum_reward
    def _map_action(self, action):
        return {
            'stim_type': 'temporal_interference',
            'freq1': action[0],
            'freq2': action[1],
            'amp1': action[2],
            'amp2': action[3],
            'total_time': 500.0, # Example
            'plot_wf': False      # Example
        }
    def normalize_firing_rate(self, fr):
        return (fr - self.observation_space.low[0]) / (self.observation_space.high[0] - self.observation_space.low[0])
    def normalize_nrg(self, nrg):
        return (nrg - self.observation_space.low[1]) / (self.observation_space.high[1] - self.observation_space.low[1])


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
            ray_results = [
                # shallow neuron 1
                simulation_Pyr.remote(
                    num_electrode=1,
                    amp1 = amplitude1, amp2 = 0,
                    freq1 = frequency1, freq2 = 0,
                    total_time = total_time,
                    plot_waveform=plot_waveform # Set to True to plot injected current
                ),
                # shallow neuron 2
                simulation_Pyr.remote(
                    num_electrode=1,
                    amp1=0, amp2 = amplitude2,
                    freq1=0, freq2=frequency2,
                    total_time=total_time,
                    plot_waveform=plot_waveform # Set to True to plot injected current
                ),
                # deep neuron
                simulation_Pyr.remote(
                    num_electrode=1,
                    amp1=amplitude1, amp2 = amplitude2,
                    freq1=frequency1, freq2=frequency2,
                    total_time=total_time,
                    plot_waveform=plot_waveform # Set to True to plot injected current
                ),
            ]
            (o_s1, ot_s1, i_s1, it_s1), (o_s2, ot_s2, i_s2, it_s2), (o_s3, ot_s3, i_s3, it_s3) = ray.get(ray_results)
            results = {
                's1_output': o_s1,
                's1_output_time': ot_s1,
                's1_input': i_s1,
                's1_input_time': it_s1,
                's2_output': o_s2,
                's2_output_time': ot_s2,
                's2_input': i_s2,
                's2_input_time': it_s2,
                'd_output': o_s3,
                'd_output_time': ot_s3,
                'd_input': i_s3,
                'd_input_time': it_s3,
            }
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
