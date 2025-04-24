import numpy as np
from agent_util import get_config
import importlib as imp
import agent
imp.reload(agent)
from agent import OffPolicy_PPOAgent
from environment import HodgkinHuxley_Environment

train_cell_ids = [8, 10, 7]
test_cell_ids = [6, 9]

# train
hh_env = HodgkinHuxley_Environment(algo='ppo')
config_file_path = 'RL_CONFIG/test_ppo.yaml' # test refers to pilot debug
config = get_config(config_file_path)
trainer = OffPolicy_PPOAgent(config)
for cell_id in train_cell_ids: # 3 * 3 * 10 = 60
    hh_env.set_cell_id(cell_id)
    trainer.train(hh_env)
    hh_env.storage.save(algo='ppo', postfix='train-1060')

# test
test_agent = OffPolicy_PPOAgent.load('latest')
for cell_id in test_cell_ids:
    hh_env.set_cell_id(cell_id)
    test_agent.play(hh_env, max_ep_len=40, num_episodes=4) # 2 * 10 * 4 = 80
    hh_env.storage.save(algo='ppo', postfix='test-1060')
