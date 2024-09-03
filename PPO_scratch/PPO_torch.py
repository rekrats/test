import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.categorical as Categorical


class PPO_memory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.values = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, action, reward, prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.values = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.rewards), \
                np.array(self.probs), \
                np.array(self.values), \
                np.array(self.dones), \
                batches