import numpy as np
import random
import torch
from collections import defaultdict, deque # can be optimized further with deque
from agent_util import init_orthogonal_weights, set_seed, data_iterator


class Critic(torch.nn.Module):
    """
    Essentially the state-value function.
    Forward function takes state input and outputs expected reward.
    """
    def __init__(self, config, device):
        super().__init__()
        self.is_cont = config.env.is_cont
        self.device = device
        self.m = torch.nn.Sequential(
            torch.nn.Linear(config.env.state_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )
        self.apply(init_orthogonal_weights)
    def forward(self, state): # gets the expected reward for a state
        return self.m(state)


class Actor(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.is_cont = config.env.is_cont
        self.device = device
        self.action_dim = config.env.action_dim
        self.action_std = config.env.action_std
        self.action_var = torch.full((self.action_dim, ), self.action_std**2).to(self.device)
        self.m = torch.nn.Sequential(
            torch.nn.Linear(config.env.state_dim, 64),
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
    def finish(self, next_state, next_value, next_state_done):
        trajectory = {
            k: torch.stack(v)
                if isinstance(v[0], torch.Tensor)
                else torch.from_numpy(np.stack(v)).to(self.device)
            for k, v in self.temp_memory.items()
        }
        steps, num_envs = trajectory['reward'].size() # num_envs should be 1
        trajectory['state'] = torch.cat(
            [trajectory['state'], torch.from_numpy(next_state).to(self.device).unsqueeze(0)],
            dim=0
        )
        trajectory['value'] = torch.cat(
            [trajectory['value'], next_value.unsqueeze(0)],
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
    def priority_sample(self, num_of_trajectories, network, inverse=0):
        segment = self.tree.total(inverse) / num_of_trajectories
        batch = defaultdict(list)
        inds = []
        for i in range(num_of_trajectories):
            a, b, s, data_idx, data = None, None, None, None, None
            try:
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                data_idx, _, data = self.tree.get(s, inverse)
                data = {k: t.clone().detach() for k, t in data.items()}
                for k, v in data.items():
                    batch[k].append(v.unsqueeze(1))
                inds.append(data_idx)
            except:
                print(f"total: {self.tree.total(inverse)} | s: {s} | segment: {segment} | [a, b]: {a}, {b} | {data_idx}, {data}")
        return self.calculate(network, batch), inds
    def update_priority(self, network, inds):
        pass
class OffPolicy_PPOAgent:
    def __init__(self, config):
        """
        config contains hyper parameters
        """
        self.config = config
        self.device = torch.device("cpu") # macbook pro
        set_seed(self.config.seed)

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
        self.timesteps = 0
    def save(self):
        # save function?
        pass

    def train(self, envs):
        # start training time?
        reward_queue = deque([0], maxlen=self.hyper_params['average_interval'])
        duration_queue = deque([0], maxlen=self.hyper_params['average_interval'])
        episodic_reward, duration, done = np.zeros(1), np.zeros(1), np.zeros(1)
        best_score = -1e9
        self.memory = PPO_Memory(
            gamma=self.hyper_params['gamma'],
            tau=self.hyper_params['tau'],
            advantage_type=self.hyper_params['advantage_type'],
            device=self.device,
            off_policy_buffer_size=self.hyper_params['off_policy_buffer_size'] # double check?
        )
        next_action_std_decay_step = self.hyper_params['network']['decay_freq']
        state, _ = envs.reset() # initial states # change
        #done = np.zeros(self.hyper_params.num_envs) # for vectorized more efficient training
        # (running multiple environments in parallel) for our purposes num_envs=1
        print("Start trainig...")
        self.policy.eval()
        # main training loop for predefined total training timesteps (hard stop)
        while self.timesteps < self.hyper_params['total_timesteps']:
            # number of episodes to run on neuron
            next_state = None
            for t in range(0, self.hyper_params['max_episode_len']): # train param
                with torch.no_grad():
                    # add observation normalizer?
                    _state = torch.from_numpy(state).to(self.device, dtype=torch.float)
                    action, logprobs, _ = self.policy(_state)
                    values = self.critic(_state)
                    values = values.flatten()
                next_state, reward, terminated, truncated, _ = envs.stimulate() # fix
                self.timesteps += 1 # finish 1 time step
                episodic_reward += reward
                duration += 1
                # add reward scaler?
                self.memory.store(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    value=values,
                    logprob=logprobs
                )
                done = terminated + truncated
                for idx, d in enumerate(done):
                    if d:
                        reward_queue.append(episodic_reward[idx])
                        duration_queue.append(duration[idx])
                        episodic_reward[idx] = 0
                        duration[idx] = 0
                state = next_state

            # Calculate gae for optimization
            with torch.no_grad():
                # observation normalizer
                next_value = self.critic(torch.Tensor(next_state).to(self.device))
                next_value = next_value.flatten()
            self.optimize(next_state, next_value, done)
            if self.hyper_params['is_cont']:
                while self.timesteps > next_action_std_decay_step:
                    next_action_std_decay_step += self.hyper_params['action_std_decay_freq'] # network param
                    self.policy.action_decay(
                        self.hyper_params['action_std_decay_rate'], # network params
                        self.hyper_params['min_action_std'] # network params
                    )
            if self.hyper_params['scheduler']:
                self.scheduler1.step()
                self.scheduler2.step()
            # logging?
            avg_score = np.round(np.mean(reward_queue), 4)
            # Writing for tensorboard?
            if avg_score >= best_score:
                #self.save('best', envs)
                best_score = avg_score



    def optimize(self, next_state, next_value, done):
        self.copy_network_param()
        self.memory.finish(next_state, next_value, done)
        fraction = self.hyper_params('fraction') # training param
        self.policy.train()
        # PPO training loop
        for _ in range(self.hyper_params['optim_epochs']): # training param
            avg_policy_loss, avg_entropy_loss, avg_value_loss = 0, 0, 0
            if 1 - fraction > 0: # uniform sample
                data, inds = self.memory.uniform_sample(
                    int(1 * (1-fraction)),
                    (self.old_policy, self.critic)
                ) # num_envs
                data = self.prepare_data(data)
                data_loader = data_iterator(self.hyper_params['batch_size'], data) # training param
                value_loss = self.optimize_critic(data_loader)
                avg_value_loss += value_loss
                if self.hyper_params['off_policy_buffer_size'] > 0:
                    self.memory.update_priority((self.old_policy, self.critic), inds)
                data_loader = data_iterator(self.hyper_params['batch_size'], data) # training param
                p_loss, e_loss = self.optimize_actor(data_loader)
                avg_policy_loss += p_loss
                avg_entropy_loss += e_loss
            if fraction > 0: # critic prioritized sampling
                data, inds = self.memory.priority_sample(
                    int(1 * fraction),
                    (self.old_policy, self.critic)
                )
                data = self.prepare_data(data)
                data_loader = data_iterator(self.hyper_params['batch_size'], data) # training params
                p_loss, e_loss = self.optimize_actor(data_loader)
                avg_policy_loss += p_loss
                avg_entropy_loss += e_loss
            # Recording / write functions?

    def play(self, env, max_ep_len, num_episodes=10):
        rewards = []
        durations = []
        for episode in range(num_episodes):
            episodic_reward = 0
            duration = 0
            state, _ = env.reset()
            for t in range(max_ep_len): # collect trajectories
                with torch.no_grad():
                    action, _, _, = self.policy(torch.Tensor(state).unsqueeze(0).to(self.device))
                next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze(0)) # adjust for simulation
                done = terminated + truncated
                episodic_reward += reward
                duration += 1
                if done:
                    break
                state = next_state
            rewards.append(episodic_reward)
            durations.append(duration)
        avg_reward = np.mean(rewards)
        avg_duration = np.mean(durations)
        # close environment

    def copy_network_param(self):
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.set_action_std(self.policy.action_std)
    def prepare_data(self, data):
        state = data['state'].float()
        action= data['action']
        logprob = data['logprob'].float()
        value_target = data['v_target'].float()
        advant = data['advant'].float()
        value = data['value'].float()
        return state, action, logprob, advant, value_target, value
    def optimize_critic(self, data_loader):
        value_losses = []
        c1 = self.hyper_params['coef_value_function'] # training param
        for batch in data_loader:
            bev_op, _, _, _, bev_vtarg, bev_v = batch
            cur_v = self.critic(bev_op)
            cur_v = cur_v.reshape(-1)
            if self.hyper_params['value_clipping']: # value loss + training param
                cur_v_clipped = bev_v + torch.clamp(
                    cur_v - bev_v,
                    -self.hyper_params['eps_clip'], # trainig params
                    self.hyper_params['eps_clip'], # training params
                )
                vloss1 = (cur_v - bev_vtarg) ** 2
                vloss2 = (cur_v_clipped - bev_vtarg) ** 2
                vf_loss = torch.max(vloss1, vloss2)
            else:
                vf_loss = (cur_v - bev_vtarg) ** 2
            vf_loss = 0.5 * vf_loss.mean()
            self.critic_optimizer.zero_grad()
            critic_loss = c1 * vf_loss
            critic_loss.backward()
            if self.hyper_params['clipping_gradient']: # training params
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()
            # record training loss data?
        return np.mean(value_losses)

    def optimize_actor(self, data_loader):
        policy_losses = []
        entropy_losses = []
        c2 = self.hyper_params['coef_entropy_penalty'] # training params
        for batch in data_loader:
            bev_ob, bev_ac, bev_logp, bev_adv, _, _ = batch
            bev_adv = (bev_adv - bev_adv.mean()) / (bev_adv.std() + 1e-7)
            _, cur_logp, cur_ent = self.policy(bev_ob, action=bev_ac)
            with torch.no_grad():
                _, old_logp, _ = self.old_policy(bev_ob, action=bev_ac)
            # policy loss
            ratio = torch.exp(cur_logp - bev_logp)
            surr1 = ratio * bev_adv
            if self.hyper_params['loss_type']: # training params
                lower = (1 - self.hyper_params['eps_clip']) * torch.exp(old_logp - bev_logp)
                upper = (1 + self.hyper_params['eps_clip']) * torch.exp(old_logp - bev_logp)
                clipped_ratio = torch.clamp(ratio, lower, upper)
                surr2 = clipped_ratio * bev_adv
                policy_surr = torch.min(surr1, surr2)
            elif self.hyper_params['loss_type'] == 'kl': # kl-divergence loss
                policy_surr = surr1 - 0.01 * torch.exp(bev_logp) * (bev_logp - cur_logp)
            else: # simple ratio loss
                policy_surr = surr1
            policy_surr = -policy_surr.mean() # policy loss
            policy_ent = -cur_ent.mean() # entropy loss
            self.policy_optimizer.zero_grad()
            policy_loss = policy_surr + c2 * policy_ent
            policy_loss.backward()
            if self.hyper_params['clipping_gradient']: # training params
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.policy_optimizer.step()
            policy_losses.append(policy_surr.item())
            entropy_losses.append(policy_ent.item())
        return np.mean(policy_losses), np.mean(entropy_losses)
