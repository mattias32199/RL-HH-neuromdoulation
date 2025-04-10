# %% imports
from matplotlib import pyplot as plt
from pointElec_simulation import pointElec_simulation
from scipy.signal import find_peaks
import numpy as np
import pandas as pd

'''
Code is meant to be run in vscode or another ide with a python kernel (ipykernel).
I have created a /simulation/outputs folder to store simulation data and logs.
The code may throw an error if you do not have this folder.
'''

def plot_membrane_potential_over_time(t, membrane_potential, cell_type, pulse_width, probe=0):
    membrane_potential_idx = find_peaks(membrane_potential, prominence=40)
    plt.figure()
    plt.title(f'Membrane potential over time ({cell_type}, pulseW={pulse_width}, probe{probe}) ')
    plt.xlabel('time (ms)')
    plt.ylabel('membrane potential (mV)')
    plt.plot(t, membrane_potential, c='royalblue')
    plt.plot(
        t[membrane_potential_idx[0]], membrane_potential[membrane_potential_idx[0]],
        'o', c='orangered', label='Detected peaks'
    )
    plt.legend(loc=4)
    plt.show()

def plot_sim_firing_rates(t, membrane_potential, cell_type, period, pulse_width):
    membrane_potential_idx = find_peaks(membrane_potential, prominence=40)
    df = pd.DataFrame({
        'time': t[membrane_potential_idx[0]],
        'membrane_pot_peaks': membrane_potential[membrane_potential_idx[0]]
    })
    df['dt_peaks'] = df.time.diff()
    df['firing_rate'] = 1 / df.dt_peaks
    plt.figure()
    plt.title(f'Firing rate across entire simulation ({cell_type}, pulseW={pulse_width})')
    plt.xlabel('time (ms)')
    plt.ylabel('firing rate (ms)')

    filter = period - pulse_width
    if (cell_type == 'Pyr') and (pulse_width > (period / 2)):
        filter = period - pulse_width + (period * 0.1)
    elif (len(membrane_potential_idx[0])) <= 4:
        print('yo')
        filter = 1000 # arbitrary large number

    plt.plot(df.time[df.dt_peaks < filter], df.firing_rate[df.dt_peaks < filter], 'o-', c='royalblue')
    plt.show()
    return df

def calc_num_firings(midpoint, pulse_width, cell_type):
    mbp, t = pointElec_simulation(
        num_electrode=1,
        amplitude=midpoint,
        pulse_width=pulse_width,
        period=100,
        total_time=100,
        cell_type=cell_type,
        plot_neuron_with_electrodes=False
    )
    mbp_idx = find_peaks(mbp[0], prominence=40)
    num_firings = len(mbp_idx[0])
    return num_firings, t, mbp

def minA_for_spike(pulse_width, cell_type, lower_bound=0.0, upper_bound=300.0, n=0):
    if n == 0:
        with open('outputs/hwk1-minA.txt', 'w') as f:
            f.write(f'--start {pulse_width}--\n')
    if n >= 100:
        print(lower_bound, upper_bound, n)
        return None, None, None
    midpoint = (lower_bound + upper_bound) / 2 # amplitude to be tested
    # run initial simulation
    num_firings, t, mbp = calc_num_firings(midpoint, pulse_width, cell_type)

    if num_firings > 1:
        upper_bound = midpoint
    elif num_firings == 0:
        lower_bound = midpoint
    else: # when num_firings == 1
        check_firings, t, mbp = calc_num_firings(midpoint-0.1, pulse_width, cell_type)
        if check_firings == 1:
            upper_bound = midpoint
        else:
            print(f'for pw={pulse_width}, A_min is {midpoint}')
            with open('outputs/hwk1-minA.txt', 'a') as f:
                f.write("--end--\n")
            return midpoint, mbp, t # should be final return
    with open('outputs/hwk1-minA.txt', 'a') as f:
        f.write(f"[{lower_bound}, {upper_bound}], {midpoint}, {num_firings}\n")
    return minA_for_spike(pulse_width, cell_type, lower_bound, upper_bound, n+1)






# %% Problem 1 + PV
'''
Problem 1
1) Plot the membrane potential over time
2) Use scipy find_peaks (prominence=40) to detect spikes
    - prominence measures how much a signal stands out from the surrounding baseline
3) calculate firing rate (# of spikes per second)
4) plot firing rate over the simulation duration
    - free to choose bin size
* 2 plots x 2 neurons = 4 plots in total
'''
num_electrode = 1 # number of electrodes surrounding the neuron
amplitude = 300 # amplitude of the square wave (uA)
pulse_width = 50 # duration of the pulse
period = 100 # period of the signal (ms)
total_time = 400 # total length of the signal (ms)
cell_type = 'PV' # type of neuron being stimulated: 'Pyr' or 'PV'

membrane_potential, t = pointElec_simulation(
    num_electrode=num_electrode,
    amplitude=amplitude,
    pulse_width=pulse_width,
    period=period,
    total_time=total_time,
    cell_type=cell_type,
    plot_neuron_with_electrodes=True
)

# %% Plots
plot_membrane_potential_over_time(t, membrane_potential[0], cell_type, pulse_width)
df = plot_sim_firing_rates(t, membrane_potential[0], cell_type, period, pulse_width)


# %% Problem 2
'''
Problem 2
1) Plot the membrane potential over time
    - A = 300uA
    - pulse_width [20, 50, 80] ms
2) Plot firing rates
3) Describe any qualitative observations
* 2 plots x 2 neurons x 3 pulse_widths = 12 plots in total
'''
num_electrode = 1 # number of electrodes surrounding the neuron
amplitude = 300 # amplitude of the square wave (uA)
pulse_width = 50 # duration of the pulse
period = 100 # period of the signal (ms)
total_time = 400 # total length of the signal (ms)
cell_type = 'Pyr' # type of neuron being stimulated: 'Pyr' or 'PV'
pulse_widths = [20, 50, 80]

data_dict = {}
for pw in pulse_widths:
    print(f'---------{pw}----------')
    key = str(pw)
    data_dict[pw] = {}
    data_dict[pw]['mbp'], data_dict[pw]['t'] = pointElec_simulation(
        num_electrode=num_electrode,
        amplitude=amplitude,
        pulse_width=pw,
        period=period,
        total_time=total_time,
        cell_type=cell_type,
        plot_neuron_with_electrodes=False
    )


# %%
for pw in pulse_widths:
    t = data_dict[pw]['t']
    membrane_potential = data_dict[pw]['mbp']
    plot_membrane_potential_over_time(t, membrane_potential[0], cell_type, pulse_width=pw)
    df = plot_sim_firing_rates(t, membrane_potential[0], cell_type, period, pulse_width=pw)
# %%
print('*', period-pulse_widths[2])
print('**', 23.2 + 22.5 + 22.55 + 33.375)
print('***', period - pulse_widths[2] + period*0.1)
df

# %%
'''
Problem 3
1) Calculate the minimum amplitude (A_min) of the square pulse that elicits a spike
    - pulse_width = [10,...,100], total_time = 100, period = 100
    - minA should be +/- 2uA
2) Plot the strength duration curve
    - x-axis: pulse_width, y-axis: A_min
* 1 plot x 2 neurons = 2 plots in total
'''
cell_type = 'Pyr' # type of neuron being stimulated: 'Pyr' or 'PV'
pulse_widths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

data_dict = {}
with open(f'outputs/hwk1-p3-{cell_type}.csv', 'w') as f:
    f.write('pulse_width, minA\n')
    for pw in pulse_widths:
        print(f'---------{pw}----------')
        key = str(pw)
        data_dict[key] = {}
        minA, t, mbp = minA_for_spike(pw, cell_type)
        data_dict[key]['minA'] = minA
        data_dict[key]['t'] = t
        data_dict[key]['mbp'] = mbp
        f.write(f'{pw}, {minA}\n')


# %%
minA_list = []
for pw in data_dict:
    if type(pw) is str:
        print(pw)
        pw = str(pw)
        minA_list.append(data_dict[pw]['minA'])

plt.figure()
plt.plot(pulse_widths, minA_list, 'o-', c='royalblue')
plt.title(f'Strength duration curve ({cell_type})')
plt.xlabel('pulse width (ms)')
plt.ylabel('mininum amplitude (uA)')
plt.show()


# %%
'''
Problem 4
1) Plot the membrane potential for num_electrode=4 for both neurons
    - A=300, pulse_width=50, period=100, total_time=400
2) Comment on the sensitivity of the neural responses to the direction of stimulation
* 4 mbp plots x 2 neurons = 8 plots in total
'''

num_electrode = 4 # number of electrodes surrounding the neuron
cell_type = 'PV' # type of neuron being stimulated: 'Pyr' or 'PV'
pulse_width = 50
period = 100

membrane_potentials, t = pointElec_simulation(
    num_electrode=num_electrode,
    amplitude=300,
    pulse_width=pulse_width,
    period=period,
    total_time=400,
    cell_type=cell_type,
    plot_neuron_with_electrodes=True
)

# %%

for i in range(len(membrane_potential)):
    plot_membrane_potential_over_time(t, membrane_potential[i], cell_type, pulse_width, probe=i)
    # df = plot_sim_firing_rates(t, mbp, cell_type, period, pulse_width)

# %%
'''
Problem 5
dx/dt = ax - y - ax(x^2 + y^2)
dy/dt = x + ay - ay(x^2 + y^2)
1) Demonstrate that the system has a limit cycle
    - convert x and y into polar coordinates
2) Find out if the limit cycle has an attractor state or repellor state
    - attractor state: stable and attracts nearby trajectories
    - repellor state: unstable and drives nearby trajectories away
'''
# cartesian plot
def dynamical_system(x, y, a=1):
    dxdt = a * x - y - a * x * (x**2 + y**2)
    dydt = x + a * y - a * y * (x**2 + y**2)
    return dxdt, dydt

initial_conditions = [2, 0]
x_list = [float(initial_conditions[0])]
y_list = [float(initial_conditions[1])]
dt = 0.01  # time step
n_max = 1000

for n in range(n_max):
    x = x_list[-1]
    y = y_list[-1]
    dxdt, dydt = dynamical_system(x, y)
    # euler's method
    x_new = x + dxdt * dt
    y_new = y + dydt * dt
    x_list.append(x_new)
    y_list.append(y_new)

plt.figure(figsize=(8, 8))
plt.plot(x_list, y_list, 'o-', c='royalblue', linewidth=1, markersize=3)
plt.xlabel('x')
plt.ylabel('y')
title = f"Plot of dynamical system with initial conditions ({x_list[0]}, {y_list[0]})"
plt.title(title)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# polar plot
def polar_dynamical_system(r):
    drdt = r * (1 - r)
    return drdt

r = np.linspace(-1, 2, 1000)
drdt = r * (1 - r)

plt.figure()
plt.plot(r, drdt, c='royalblue')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.xlabel('r')
plt.ylabel('dr/dt')
plt.title('Phase plot of system in polar domain')
plt.grid()
plt.show()
