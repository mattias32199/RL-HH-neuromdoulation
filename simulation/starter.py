from matplotlib import pyplot as plt
from pointElec_simulation import pointElec_simulation
from scipy.signal import find_peaks
import numpy as np

num_electrode = 1 # number of electrodes surrounding the neuron
amplitude = 300 # amplitude of the square wave (uA)
pulse_width = 20 # duration of the pulse
period = 100 # period of the signal (ms)
total_time = 300 # total length of the signal (ms)
cell_type = 'PV' # type of neuron being stimulated: 'Pyr' or 'PV'

membrane_potential,t = pointElec_simulation(
    num_electrode=num_electrode,
    amplitude=amplitude,
    pulse_width=pulse_width,
    period=period,
    total_time=total_time,
    cell_type=cell_type,
    plot_neuron_with_electrodes = True
)

for i in range(membrane_potential.shape[0]):
    plt.plot(t,membrane_potential[i,:])
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Soma recording')
    plt.show()
