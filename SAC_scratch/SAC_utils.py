import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

class SAC_memory:
    def __init__(self, maxsize,  input_size, num_actions) -> None:
        self.memsize = maxsize
        self.mem_cntr = 0 # counter for memory
        self.state_memory = np.zeros((self.memsize, *input_size))
        self.action_memory = np.zeros((self.memsize, num_actions))
        self.reward_memory = np.zeros(self.memsize)
        self.new_state_memory = np.zeros((self.memsize, *input_size))
        self.terminal_memory = np.zeros(self.memsize)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.memsize
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.memsize)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_d, n_actions, fc1_d=256, fc2_d=256,
                 name__='critic', ckt_dir="./SAC_scaratch./tmp/sac") -> None:
        super(CriticNetwork, self).__init__()
        self.input_d = input_d[0]
        self.n_actions = n_actions
        self.fc1_d = fc1_d
        self.fc2_d = fc2_d
        self.name = name__
        self.ckpt_file = os.path.join(ckt_dir, self.name+'_sac')

        self.fc1 = nn.Linear(self.input_d + self.n_actions, self.fc1_d)
        self.fc2 = nn.Linear(self.fc1_d, self.fc2_d)
        self.q = nn.Linear(self.fc2_d, 1) # output is Q value

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q = self.q(action_value)
        return q
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), self.ckpt_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.ckpt_file))


class Valuenetwork(nn.Module):
    def __init__(self, beta, input_d, fc1_d=256, fc2_d=256, 
                name='value', ckt_dir="./SAC_scaratch./tmp/sac") -> None:
        super(Valuenetwork, self).__init__()
        self.input_d = input_d[0]
        self.fc1_d = fc1_d
        self.fc2_d = fc2_d
        self.name = name
        self.ckpt_file = os.path.join(ckt_dir, self.name+'_sac')

        self.fc1 = nn.Linear(self.input_d, self.fc1_d)
        self.fc2 = nn.Linear(self.fc1_d, self.fc2_d)
        self.v = nn.Linear(self.fc2_d, 1) # output is V value

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)

        return v
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), self.ckpt_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.ckpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_d, maxaction, fc1_d=256, fc2_d=256, n_actions=2,
                 name='actor', ckt_dir="./SAC_scaratch./tmp/sac") -> None:
        super(ActorNetwork, self).__init__()
        self.input_d = input_d[0]
        self.max_action = maxaction
        self.action_d = n_actions
        self.fc1_d = fc1_d
        self.fc2_d = fc2_d
        self.name = name
        self.ckpt_file = os.path.join(ckt_dir, self.name+'_sac')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_d, self.fc1_d)
        self.fc2 = nn.Linear(self.fc1_d, self.fc2_d)
        self.mu = nn.Linear(self.fc2_d, self.action_d)
        self.sigma = nn.Linear(self.fc2_d, self.action_d)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(sigma, min=1e-5, max=1)
        # 之所以返回 mu 和 sigma, 而不是直接返回一个 action 向量
        # 是因为我们需要计算 action 的概率分布，然后从分布中采样，这里的 action 是连续的

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma) # 创建一个正态分布

        if reparameterize:
            actions = probabilities.rsample() # 从分布中采样一个动作
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        # tanh 函数的输出范围是 (-1, 1), 乘以 max_action 之后，输出范围是 (-max_action, max_action)
        log_probs = probabilities.log_prob(actions)
        # print('log_probs:', log_probs)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))