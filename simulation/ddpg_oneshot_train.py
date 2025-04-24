import numpy as np
from agent_util import get_config
from agent import OffPolicy_PPOAgent
from environment import HodgkinHuxley_Environment
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

train_cell_ids = [8, 10, 7]
test_cell_ids = [6, 9]

hh_ddpg_env = HodgkinHuxley_Environment(algo='ddpg')
n_actions = hh_ddpg_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG("MlpPolicy", hh_ddpg_env, action_noise=action_noise, verbose=1)
for cell_id in train_cell_ids:
    hh_ddpg_env.set_cell_id(cell_id)
    model.set_env(hh_ddpg_env)
    model.learn(total_timesteps=1000) # 3 episides * 30 = 90
    hh_ddpg_env.storage.save(algo='ddpg', postfix='train-1060')
model.save("ddpg-train") # save training model

test_model = DDPG.load("ddpg-train")
for cell_id in test_cell_ids:
    hh_ddpg_env.set_cell_id(cell_id)
    for episode in range(4): # 2 neuron * 4 episodes * max_10_timesteps = 80 timesteps
        state, _ = hh_ddpg_env.reset()
        terminated = False
        total_reward = 0
        iter = 0
        while (not terminated) and (iter < 40):
            action, _states = test_model.predict(state)
            state, reward, terminated, truncated, info = hh_ddpg_env.step(action)
            iter += 1
    hh_ddpg_env.storage.save(algo='ddpg', postfix='test-1060')
