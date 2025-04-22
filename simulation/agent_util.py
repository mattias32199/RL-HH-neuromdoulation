import torch
import numpy as np
import random
from omegaconf import OmegaConf


##############################################
####### UTIL FUNCTIONS USED BY RL AGENT ######
##############################################

def orthogonal_init(tensor, gain=1):
    """
    https://github.com/implementation-matters/code-for-paper/blob/094994f2bfd154d565c34f5d24a7ade00e0c5bdb/src/policy_gradients/torch_utils.py#L494
    Fills the input Tensor using the orthogonal initialization scheme from OpenAI
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)
    if rows < cols:
        flattened.t_()
    u, s, v = torch.svd(flattened, some=True) # Compute the qr factorization
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

def init_normal_weights(m):
    """
    Not used.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.1)
        torch.nn.init.constant_(m.bias, 0.1)

# Used by Agent + Critic
def init_orthogonal_weights(m):
    """
    Wrapper of OpenAI's orthogonal init.
    """
    if isinstance(m, torch.nn.Linear):
        orthogonal_init(m.weight)
        torch.nn.init.constant_(m.bias, 0.1)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def data_iterator(batch_size, given_data, t=False):
    # Simple mini-batch splitter
    ob, ac, oldpas, adv, tdlamret, old_v = given_data
    total_size = len(ob)
    inds = np.arange(total_size)
    np.random.shuffle(inds)
    n_batches = total_size // batch_size
    for nb in range(n_batches):
        idx = inds[batch_size * nb : batch_size * (nb + 1)]
        yield ob[idx], ac[idx], oldpas[idx], adv[idx], tdlamret[idx], old_v[idx]

def get_config(yaml_file=None, yaml_string=None, **kwargs):
    assert yaml_file is not None or yaml_string is not None, 'Must enter yaml_file or string'
    if yaml_string is not None:
        conf = OmegaConf.create(yaml_string)
    else:
        conf = OmegaConf.load(yaml_file)
    if kwargs:
        conf.update(kwargs)
    if 'checkpoint_path' not in conf:
        conf.checkpoint_path = '.'
    return conf
