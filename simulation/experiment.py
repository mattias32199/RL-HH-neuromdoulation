# %%
import numpy as np
from agent_util import get_config
from agent import OffPolicy_PPOAgent
from environment import HodgkinHuxley_Environment

# %%
# TRAIN
config_path = ""
config = get_config(config_path)
trainer = OffPolicy_PPOAgent(config)
env = HodgkinHuxley_Environment()

# %%
# PLAY



# %%
# DDPG 1
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

hh_ddpg_env = HodgkinHuxley_Environment()
n_actions = hh_ddpg_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = DDPG("MlpPolicy", hh_ddpg_env, action_noise=action_noise, verbose=1)
# %%
# DDPG 2
model.learn(total_timesteps=30)

# from stable_








# %%
def parse_args():
    parser.add_argument("--config", type=str)
    parser.add_argument("--name", type=str, default=None)
