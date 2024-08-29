import flappy_bird_gymnasium
import gymnasium
from Experience_Replay import ReplayMemory

import torch
import itertools
import yaml

from dqn import DQN

class Agent:
    def __init__(self, hyperparameters_set: str):
        with open(hyperparameters_set, "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameters_set]
            # print(hyperparameters)
        
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]   
        self.epsilon_min = hyperparameters["epsilon_min"] 

    def run(self, is_training: bool = True, render: bool = False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        state_dim = now_state.shape[0]
        action_dim = env.action_space.n
        policy_dqn = DQN(state_dim, action_dim)

        if is_training:
            memory = ReplayMemory(max_len=self.replay_memory_size)

        reward_list = []

        for episode in itertools.count():
            now_state = env.reset()
            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Next action:
                action = env.action_space.sample()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)
                episode_reward += reward

                if is_training:
                    memory.append((now_state, action, reward, new_state, terminated))

                now_state = new_state
            
            reward_list.append(episode_reward)



