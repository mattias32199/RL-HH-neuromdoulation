import numpy as np
from agent_util import get_config
from agent import OffPolicy_PPOAgent
from environment import HodgkinHuxley_Environment
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

hh_ddpg_env = HodgkinHuxley_Environment()
n_actions = hh_ddpg_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG("MlpPolicy", hh_ddpg_env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=30)
