# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 18:25:18 2025

@author: U2019
"""

# sac_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in np.random.choice(len(self.buffer), size=batch_size)])
        return np.array(states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(next_states), np.array(dones).reshape(-1, 1)

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3_mean = nn.Linear(256, action_dim)
        self.l3_log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.l3_mean(x)
        log_std = self.l3_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

class SACAgent:
    def __init__(self, config):
        params = config["sac_agent_params"]
        self.device = torch.device(params["device"])
        self.state_dim = params["state_dim"]
        self.action_dim = params["action_dim"]
        
        # 将action_bounds字典转换为tensor
        action_bounds_dict = params["action_bounds"]
        self.action_scale = torch.tensor([(bounds[1] - bounds[0]) / 2.0 for bounds in action_bounds_dict.values()], dtype=torch.float32).to(self.device)
        self.action_bias = torch.tensor([(bounds[1] + bounds[0]) / 2.0 for bounds in action_bounds_dict.values()], dtype=torch.float32).to(self.device)
        
        self.actor = Actor(self.state_dim, self.action_dim, 1.0).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params["actor_lr"])

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params["critic_lr"])

        self.replay_buffer = ReplayBuffer(params["buffer_size"])
        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.alpha = params["alpha"]
        self.batch_size = params["batch_size"]
        self.corrected_policy = bool(params.get("corrected_policy", False))
        self.learn_updates = 0

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        # 将-1到1的动作映射到实际的gamma参数范围
        scaled_action = action * self.action_scale + self.action_bias
        self.last_action = {"z": z.detach().cpu().numpy().flatten(), "u": action.detach().cpu().numpy().flatten(), "mapped": scaled_action.detach().cpu().numpy().flatten()}
        return scaled_action.cpu().detach().numpy().flatten()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_state)
            next_std = torch.exp(next_log_std)
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_z = next_normal.sample()
            next_u = torch.tanh(next_z)
            next_logp = next_normal.log_prob(next_z)
            if self.corrected_policy:
                next_logp = next_logp - torch.log(1 - next_u.pow(2) + 1e-6) - torch.log(self.action_scale)
            next_action_prob = next_logp.sum(1, keepdim=True)
            
            # Scale and shift the action
            next_action = torch.tanh(next_z) * self.action_scale + self.action_bias

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_action_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample() if self.corrected_policy else normal.sample()
        u = torch.tanh(z)
        logp = normal.log_prob(z)
        if self.corrected_policy:
            logp = logp - torch.log(1 - u.pow(2) + 1e-6) - torch.log(self.action_scale)
        action_prob = logp.sum(1, keepdim=True)
        
        scaled_action = torch.tanh(z) * self.action_scale + self.action_bias
        q1_pi, q2_pi = self.critic(state, scaled_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((self.alpha * action_prob) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.learn_updates += 1
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)