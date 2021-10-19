import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # 将数据加入buffer

    def sample(self, batch_size):  # 从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

    def pop(self):
        self.buffer.pop()

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        layers = [torch.nn.Linear(state_dim, hidden_dim[0]), torch.nn.ReLU()]
        for i in range(len(hidden_dim)-1):
            layers.append(torch.nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Linear(hidden_dim[-1], action_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, device, learning_rate=2e-3, gamma=0.98,
                 epsilon=0.01, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.q_net = Qnet(state_dim, self.hidden_dim, self.action_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_dim, self.hidden_dim, self.action_dim).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器，记录更新次数

    def take_action(self, state):  # epsilon greedy策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save_model(self, path, env_name):
        state = {'net': self.q_net.state_dict(),
                 'name': env_name,
                 'dim': [self.state_dim, self.hidden_dim, self.action_dim]}
        torch.save(state, os.path.join(path, state['name']+'.pth'))

    def load_model(self, state, env_name):
        if state['name'] != env_name:
            raise ValueError("Not expected model")
        self.q_net.load_state_dict(state['net'])


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class ConvQnet(torch.nn.Module):
    def __init__(self, in_channels, action_dim):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 32, (3, 2), (3, 2))
        self.conv2 = ConvBlock(32, 64, (5, 5), (5, 5))
        self.conv3 = ConvBlock(64, 64, (2, 2), (2, 2))
        self.fc = torch.nn.Linear(7 * 8 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x.float() / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(self.fc(x.view(x.size()[0], -1)))
        return self.head(x)


class ConvDQN:
    def __init__(self, state_dim, action_dim, device, learning_rate=2e-3, gamma=0.98, epsilon=0.01, target_update=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.q_net = ConvQnet(state_dim[0], self.action_dim).to(device)  # Q网络
        self.target_q_net = ConvQnet(state_dim[0], self.action_dim).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器，记录更新次数

    def take_action(self, state):  # epsilon greedy策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save_model(self, path, env_name):
        self.q_net.to(torch.device('cpu'))
        # net = self.q_net.to(torch.device('cpu'))
        state = {'net': self.q_net.state_dict(),
                 'name': env_name+'_conv',
                 'dim': [self.state_dim, self.action_dim]}
        torch.save(state, os.path.join(path, state['name']+'.pth'))
        self.q_net.to(self.device)
        # self.q_net = self.q_net.to(self.device)

    def load_model(self, state, env_name):
        if state['name'] != env_name+'_conv':
            raise ValueError("Not expected model")
        self.q_net.load_state_dict(state['net'])