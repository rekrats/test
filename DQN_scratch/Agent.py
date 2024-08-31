import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np


import random
import torch
from torch import nn
import yaml

from Experience_Replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = ".\\DQN_scratch\\runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, hyperparameters_set: str):
        with open('.\DQN_scratch\hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameters_set]
            # print(hyperparameters)
        self.hyperparameters_set = hyperparameters_set
        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameters_set}.png')

    def run(self, is_training: bool = True, render: bool = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        policy_dqn = DQN(state_dim, action_dim, self.fc1_nodes).to(device)
        

        if is_training:
            epsilon = self.epsilon_init
            epsilon_list = []
            memory = ReplayMemory(max_len=self.replay_memory_size)
            target_dqn = DQN(state_dim, action_dim, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            sync_count = 0
            best_reward = -1e9
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            print(f"Model loaded from {self.MODEL_FILE}")
            policy_dqn.eval()

        reward_list = []

        for episode in itertools.count():
            now_state, _ = env.reset()
            now_state = torch.tensor(now_state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while ((not terminated) and episode_reward < self.stop_on_reward):
                # Next action:
                if is_training:
                    if random.random() < epsilon:
                        action = env.action_space.sample()
                        action = torch.tensor(action, dtype=torch.int64, device=device)
                    else:
                        # (state_dim,) -> (1, state_dim)
                        with torch.no_grad():
                            action = policy_dqn(now_state.unsqueeze(0)).squeeze().argmax()
                else:
                    with torch.no_grad():
                        action = policy_dqn(now_state.unsqueeze(0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                episode_reward += reward

                # convert to tensor
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((now_state, action, reward, new_state, terminated))
                    sync_count += 1
                now_state = new_state

            reward_list.append(episode_reward)

            if is_training:
                epsilon_list.append(epsilon)
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                if episode_reward > best_reward:
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    best_reward = episode_reward
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                if len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                if sync_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    sync_count = 0

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_list, epsilon_list)
                    last_graph_update_time = current_time

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        now_states, actions, rewards, new_states, terminateds = zip(*mini_batch)
        now_states = torch.stack(now_states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        new_states = torch.stack(new_states)
        terminateds = torch.tensor(terminateds, dtype=torch.float, device=device)

        with torch.no_grad():
            target_q = rewards + self.discount_factor_g * target_dqn(new_states).max(dim=1)[0] * (1 - terminateds)
        
        q = policy_dqn(now_states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        
        loss = self.loss_fn(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    # Parse command line inputs
    # parser = argparse.ArgumentParser(description='Train or test model.')
    # parser.add_argument('hyperparameters', help='')
    # parser.add_argument('--train', help='Training mode', action='store_true')
    # args = parser.parse_args()

    dql = Agent(hyperparameters_set="flappybird1")

    if True:
        dql.run(is_training=True, render=False)
    else:
        dql.run(is_training=False, render=True)