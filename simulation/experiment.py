# %%
from agent_util import get_config
from agent import OffPolicy_PPOAgent
from environment import HodgkinHuxley_Environment

# TRAIN
config_path = ""
config = get_config(config_path)
trainer = OffPolicy_PPOAgent(config)
env = HodgkinHuxley_Environment()

# %%
# PLAY












# %%
def parse_args():
    parser.add_argument("--config", type=str)
    parser.add_argument("--name", type=str, default=None)
