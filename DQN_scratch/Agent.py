import flappy_bird_gymnasium
import gymnasium
from Experience_Replay import ReplayMemory

import torch
import torch.nn as nn
import itertools
import yaml
import random

from dqn import DQN

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameters_set: str):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameters_set]
            # print(hyperparameters)
        
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]   
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def run(self, is_training: bool = True, render: bool = False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        policy_dqn = DQN(state_dim, action_dim).to(device)

        if is_training:
            memory = ReplayMemory(max_len=self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(state_dim, action_dim).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            sync_count = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        reward_list = []
        epsilon_list = []

        for episode in itertools.count():
            now_state, _ = env.reset()
            now_state = torch.tensor(now_state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Next action:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # (state_dim,) -> (1, state_dim)
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

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            reward_list.append(episode_reward)
            epsilon_list.append(epsilon)
            if episode % 50 == 0:
                print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon}")

            if is_training:
                if len(memory) >= self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                if sync_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    sync_count = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        for now_state, action, reward, new_state, terminated in mini_batch:
            if terminated:
                target_q = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(new_state).max()
            q = policy_dqn(now_state)[action]
            loss = self.loss_fn(q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    agent = Agent("cartpole1")
    agent.run(is_training=True, render=False)