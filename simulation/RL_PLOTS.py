# %%
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

ddpg_train_log = pd.read_csv('RL_SAVE/ddpg-train-test.csv')
ddpg_test_log = pd.read_csv('RL_SAVE/ddpg-train-1060.csv')


ppo_train_log = pd.read_csv('RL_SAVE/ppo-train-train.csv')
ppo_test_log = pd.read_csv('RL_SAVE/ppo-train-1060.csv')

mobo_test_log = pd.read_csv('RL_SAVE/mobo-test-1060.csv')

ppo_log = pd.read_csv('RL_SAVE/ppo-test-1060.csv')
ddpg_log = pd.read_csv('RL_SAVE/ddpg-train-1060.csv')
mobo_log = pd.read_csv('RL_SAVE/mobo-test-1060.csv')


# %%
plt.figure()
plt.plot(ppo_log.index, ppo_log.reward, label='Off-policy PPO', c='royalblue')
plt.plot(ddpg_log.index, ddpg_log.reward, label='DDPG', c='orangered')
plt.plot(mobo_log.index, mobo_log.reward, label='MOBO', c='green')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.title('Reward (Training+Test)')
plt.show()


plt.figure()
plt.plot(ppo_log.index, ppo_log.cumulative_reward, label='Off-policy PPO', c='royalblue')
plt.plot(ddpg_log.index, ddpg_log.cumulative_reward, label='DDPG', c='orangered')
plt.plot(mobo_log.index, mobo_log.cumulative_reward, label='MOBO', c='green')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward (Training+Test)')
plt.show()

plt.figure()
plt.plot(ppo_log.index, ppo_log.average_reward, label='Off-policy PPO', c='royalblue')
plt.plot(ddpg_log.index, ddpg_log.average_reward, label='DDPG', c='orangered')
plt.plot(mobo_log.index, mobo_log.average_reward, label='MOBO', c='green')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Average Reward')
plt.title('Average Reward (Training+Test)')
plt.show()

plt.figure()
plt.plot(ppo_log.index, ppo_log.fr_state, label='Off-policy PPO', c='royalblue')
plt.plot(ddpg_log.index, ddpg_log.fr_state, label='DDPG', c='orangered')
plt.plot(mobo_log.index, mobo_log.fr_state, label="MOBO", c='green')
plt.axhline(90, c='gold', label='Target firing rate')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing Rate (Training + Test)')
plt.show()

plt.figure()
plt.title('Input Applied Energy (Training + Test)')
plt.plot(ppo_log.index, ppo_log.nrg_state, label='Off-policy PPO', c='royalblue')
plt.plot(ddpg_log.index, ddpg_log.nrg_state, label='DDPG', c='orangered')
plt.plot(mobo_log.index, mobo_log.nrg_state, label="MOBO", c='green')
plt.axhline(0.0004, c='gold', label='Upper bound for applied energy')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('A^2 * s')
plt.show()

print(list(ddpg_train_log))
# %%
ppo_test_log.tail()
