import numpy as np
import torch
from collections import defaultdict, deque # can be optimized further with deque
from agent_util import init_orthogonal_weights, set_seed

class Critic(torch.nn.Module):
    """
    Essentially the state-value function.
    Forward function takes state input and outputs expected reward.
    """
    def __init__(self, state_dim, is_cont, device):
        super().__init__()
        self.is_cont = is_cont
        self.device = device
        self.m = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )

        self.apply(init_orthogonal_weights)
    def forward(self, state): # gets the expected reward for a state
        return self.m(state)

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, is_cont, device, action_std):
        super().__init__()
        self.is_cont = is_cont
        self.device = device
        self.action_dim = action_dim
        self.action_std = action_std
        self.action_var = torch.full((self.action_dim, ), self.action_std**2).to(self.device)
        self.m = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, self.action_dim),
            torch.nn.Tanh(),
        )
        self.apply(init_orthogonal_weights)

    def action_decay(self, action_std_decay_rate, min_action_std):
        # Change the action variance
        if self.is_cont: # only for continuous action space
            # decrease action_std (adjusted through decay parameter)
            self.action_std = self.action_std - action_std_decay_rate
            # quantize action_std
            self.action_std = round(self.action_std, 4)
            # action_std lower_bound
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std

    def set_action_std(self, action_std):
        self.action_std = action_std
        self.action_var = torch.full((self.action_dim, ), self.action_std**2).to(self.device)

    def forward(self, state, action=None):
        if self.is_cont:
            action_mean = self.m(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.m(state)
            dist = torch.distributions.Categorical(action_probs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):
        if max_steps < warmup_steps:
            max_steps = warmup_steps + 1
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return max(1.0 - float((step - warmup_steps) / (max_steps - warmup_steps)), 0)
        # inherit class from torch.optim.lr_schedular.LambdaLR
        super(WarmupLinearScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.node = [] # [[0, 0]] * (2 * capacity - 1)
        for _ in range(2*capacity-1):
            self.node.append([0, 0])
        self.data = [None] * capacity
        self.data_idx = 0
        self.size = 0
    def total(self, inverse=0):
        return self.node[0][inverse]
    def update(self, data_idx, value, inverse=0):
        idx = data_idx + self.capacity - 1
        change = value - self.node[idx][inverse]
        self.node[idx][inverse] = 0
        parent = (idx - 1) // 2
        while parent >= 0:
            self.node[parent][inverse] += change
            parent = (parent - 1) // 2
    def add(self, value, data):
        self.data[self.data_idx] = data
        self.update(self.data_idx, 1)
        self.update(self.data_idx, 1, inverse=1)
        self.data_idx = (self.data_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s, inverse=0):
        idx = 0
        while 2 * idx + 1 < len(self.node):
            left, right = 2 * idx + 1, 2 * idx + 2
            if s <= self.node[left][inverse]:
                idx = left
            else:
                idx = right
                s -= self.node[left][inverse]
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.node[idx][inverse], self.data[data_idx]


class PPO_Memory:
    def __init__(self, gamma, tau, advantage_type, device, off_policy_buffer_size):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.advantage_type = advantage_type
        self.tree = SumTree(off_policy_buffer_size)
        self.temp_memory = {
             "state": [],
             "action": [],
             "reward": [],
             "done": [],
             "value": [],
             "logprob": []
         }
    def store(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.temp_memory:
                print("warning, wrong data insertion")
            else:
                self.temp_memory[k].append(v)
    def finish(self, next_state, next_state_value, next_state_done):
        trajectory = {
            k: torch.stack(v)
                if isinstance(v[0], torch.Tensor)
                else torch.from_numpy(np.stack(v)).to(self.device)
            for k, v in self.temp_memory.items()
        }
        steps, num_envs = trajectory['reward'].size()
        trajectory['state'] = torch.cat(
            [trajectory['state'], torch.from_numpy(next_state).to(self.device).unsqueeze(0)],
            dim=0
        )
        trajectory['value'] = torch.cat(
            [trajectory['value'], next_state_value.unsqueeze(0)],
            dim=0
        )
        trajectory['done'] = torch.cat(
            [trajectory['done'], torch.from_numpy(next_state_done).to(self.device).unsqueeze(0)],
        )
        priority = torch.ones((num_envs))
        for i in range(num_envs):
            self.tree.add(
                value=priority[i].item(),
                data={k: v[:, i] for k, v in trajectory.items()}
            )
        self.reset_temp_memory()
    def uniform_sample(self, num_of_trajectories, network):
        inds = np.arange(self.tree.size)
        np.random.shuffle(inds)
        batch = defaultdict(list)
        for i in inds:
            for k, v in self.tree.data[i].items():
                batch[k].append(v.unsqueeze[1])
        return self.calculate(network, batch), inds
    def calculate(self, network, batch):
        batch = {
            k: torch.cat(batch[k], dim=1)
                for k in ["state", "action", "reward", "done", "value", "logprob"]
        }
        if self.advantage_type=='gae':
            batch = self.calc_gae(**batch)
        elif self.advantage_type == 'vtrace':
            batch = self.calc_vtrace(**batch)
        elif self.advantage_type == 'vtrace_gae':
            batch = self.calc_vtrace_gae(**batch)
        return batch
    def calc_gae(self, state, action, reward, done, value, logprob):
        steps, num_envs = reward.size()
        gae_t = torch.zeros(num_envs).to(self.device)
        advantage = torch.zeros((steps, num_envs)).to(self.device)
        for t in reversed(range(steps)): # Each episode is calculated separately by done.
            # delta(t) = reward(t) + γ * value(t+1) - value(t)
            delta_t = reward[t] + self.gamma * value[t+1] * (1 - done[t + 1]) - value[t]
            # gae(t) = delta(t) + γ * τ * gae(t + 1)
            gae_t = delta_t + self.gamma * self.tau * gae_t * (1 - done[t + 1])
            advantage[t] = gae_t
        v_target = advantage + value[:steps] # Remove value in the next state
        trajectory = {
            "state" : state,
            "reward" : reward,
            "action" : action,
            "logprob" : logprob,
            "done" : done,
            "value" : value,
            "advant" : advantage,
            "v_target" : v_target
        }
        # The first two values ​​refer to the trajectory length and number of envs.
        return {k: v[:steps].reshape(-1, *v.size()[2:]) for k, v in trajectory.items()}
    def calc_vtrace(self, network, state, action, reward, done, value, logprob):
        actor, critic = network
        steps, num_envs = reward.size()
        with torch.no_grad():
            values = critic(state.reshape((steps + 1) * num_envs, *state.size()[2:]).float())
            _, pi_logp, _ = actor(
                state[:steps].reshape(steps * num_envs, *state.size()[2:]).float(),
                action=action.reshape(steps * num_envs, *action.size()[2:])
            )
            values  = values.reshape(steps + 1, num_envs)
            pi_logp = pi_logp.reshape(steps, num_envs)
            ratio   = torch.exp(pi_logp - logprob)
        c = torch.min(torch.ones_like(ratio), ratio) # c
        rho = torch.min(torch.ones_like(ratio), ratio) # rho
        vtrace = torch.zeros((steps + 1, num_envs)).to(self.device)
        for t in reversed(range(steps)):
            # delta(s) = rho * (reward(s) + γ * Value(s+1) - Value(s))
            delta = rho[t] * (reward[t] + self.gamma * value[t + 1] * (1 - done[t + 1]) - value[t])
            # vtrace(s) = delta(s) + γ * c_s * vtrace(s+1)
            vtrace[t] = delta + self.gamma * c[t] * vtrace[t + 1] * (1 - done[t + 1])
        vtrace = vtrace + value # vtrace(s) = vtrace(s) + value(s)
        v_target = vtrace
        advantage = rho * (reward + self.gamma * vtrace[1:] - value[:steps])
        trajectory = {
            "state"     : state,
            "reward"    : reward,
            "action"    : action,
            "logprob"   : logprob,
            "done"      : done,
            "value"     : value,
            "advant"    : advantage,
            "v_target"  : v_target
        }
        # The first two values ​​refer to the trajectory length and number of envs.
        return {k: v[:steps].reshape(-1, *v.size()[2:]) for k, v in trajectory.items()}
    def calc_vtrace_gae(self, network, state, action, reward, done, value, logprob):
        actor, critic = network
        steps, num_envs = reward.size()
        with torch.no_grad():
            values = critic(state.reshape((steps + 1) * num_envs, *state.size()[2:]).float())
            _, pi_logp, _ = actor(
                state[:steps].reshape(steps * num_envs, *state.size()[2:]).float(),
                action=action.reshape(steps * num_envs, *action.size()[2:])
            )
            values  = values.reshape(steps + 1, num_envs)
            pi_logp = pi_logp.reshape(steps, num_envs)
            ratio = torch.exp(pi_logp - logprob)
            ratio = torch.min(torch.ones_like(ratio), ratio)
        gae_t = torch.zeros(num_envs).to(self.device)
        advantage = torch.zeros((steps, num_envs)).to(self.device)
        for t in reversed(range(steps)): # Each episode is calculated separately by done.
            # delta(t)   = rho(t) * (reward(t) + γ * value(t+1) - value(t))
            delta_t = reward[t] + self.gamma * value[t+1] * (1 - done[t+1]) - value[t]
            delta_t = ratio[t] * delta_t
            # gae(t)     = delta(t) + rho(t) * γ * τ * gae(t + 1)
            gae_t = delta_t + ratio[t] * self.gamma * self.tau * gae_t * (1 - done[t + 1])
            advantage[t] = gae_t
        v_target = advantage + value[:steps] # Remove value in the next state
        trajectory = {
            "state"     : state,
            "reward"    : reward,
            "action"    : action,
            "logprob"   : logprob,
            "done"      : done,
            "value"     : value,
            "advant"    : advantage,
            "v_target"  : v_target
        }
        # The first two values ​​refer to the trajectory length and number of envs.
        return {k: v[:steps].reshape(-1, *v.size()[2:]) for k, v in trajectory.items()}
    def reset_temp_memory(self):
        self.temp_memory = {k: [] for k, v in self.temp_memory.items()}


class OffPolicy_PPOAgent:
    def __init__(self, hyper_params):
        self.device = torch.device("cpu") # macbook pro
        set_seed(hyper_params['seed'])

        self.policy = Actor(
            state_dim=hyper_params['state_dim'],
            action_dim=hyper_params['action_dim'],
            is_cont=hyper_params['is_cont'],
            action_std=hyper_params['action_std'],
            device=self.device,
        ).to(self.device)
        self.old_policy = Actor(
            state_dim=hyper_params['state_dim'],
            action_dim=hyper_params['action_dim'],
            is_cont=hyper_params['is_cont'],
            action_std=hyper_params['action_std'],
            device=self.device,
        ).to(self.device)
        self.critic = Critic(
            state_dim=hyper_params['state_dim'],
            is_cont=hyper_params['is_cont'],
            device=self.device,
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            **hyper_params['network']
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            **hyper_params['network']
        )

        if hyper_params['scheduler']:
            self.scheduler1 = WarmupLinearScheduler(
                optimizer=self.policy_optimizer,
                warmup_steps=0, # look at this again?
                max_steps = hyper_params['max_steps'] # look at this again too!!
            )
            self.scheduler2 = WarmupLinearScheduler(
                optimizer=self.critic_optimizer,
                warmup_steps=0, # !!
                max_steps = hyper_params['max_steps'] # !!
            )

        # reward scaler?
        # observation scaler?
        # implement logging?

        self.hyper_params = hyper_params

    def save(self):
        # save function?
        pass

    def train(self):

        # start training time?


        pass
