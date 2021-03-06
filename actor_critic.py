import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import numpy as np


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, action_dim)
        self.activ1 = torch.nn.LeakyReLU()
        self.activ2 = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.activ1(self.fc1(x))
        x = self.activ2(self.fc2(x))
        return 2*torch.sigmoid(self.fc3(x))-1


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.activ1 = torch.nn.LeakyReLU()
        self.activ2 = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.activ1(self.fc1(x))
        x = self.activ2(self.fc2(x))
        return self.fc3(x)


class ActorCritic:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma):
        self.gamma = gamma
        self.actor = PolicyNet(state_dim, action_dim)
        self.critic = ValueNet(state_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.action_dim = action_dim
        self.state_dim = state_dim

    def take_action(self, state):
        state = torch.FloatTensor([state])
        action = self.actor(state)
        cov_matrix = torch.eye(self.action_dim)
        distb = torch.distributions.MultivariateNormal(action, covariance_matrix=cov_matrix)
        return distb.sample()

    def get_log_prob(self, state, actions):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        cov_matrix = torch.eye(self.action_dim)
        distb = torch.distributions.MultivariateNormal(action, covariance_matrix=cov_matrix)
        log_prob = distb.log_prob(actions)

        return log_prob

    def update(self, transition_dict):
        states = torch.FloatTensor(transition_dict['states'])
        actions = torch.FloatTensor(transition_dict['actions'])
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1)
        next_states = torch.FloatTensor(transition_dict['next_states'])
        dones = torch.FloatTensor(transition_dict['dones']).view(-1)

        td_target = rewards + self.gamma * self.critic(next_states).view(-1) * (1 - dones)
        td_delta = td_target - self.critic(states).view(-1)
        log_probs = self.get_log_prob(states, actions.view(-1, 1)).view(-1)
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states).view(-1), td_target.detach()))
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()

    def save_model(self, path, env_name):
        self.actor.to(torch.device('cpu'))
        self.critic.to(torch.device('cpu'))
        state = {'actor': self.actor.state_dict(),
                 'critic': self.critic.state_dict(),
                 'name': env_name,
                 'dim': [self.state_dim, self.action_dim]}
        torch.save(state, os.path.join(path, state['name']+'.pth'))

    def load_model(self, state, env_name):
        if state['name'] != env_name:
            raise ValueError("Not expected model")
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])








