import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.categorical as Categorical

import os


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


class PPO_actornet(nn.Module):
    def __init__(self, n_actions, input_d, alpha, fc1=256, fc2=256, ckp_dir="./PPO_scratch/tmp/ppo"):
        super(PPO_actornet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_d, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.checkpoint_file = os.path.join(ckp_dir, "actor_torch_ppo")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical.Categorical(dist) # 生成一个概率分布
        return dist
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PPO_criticnet(nn.Module):
    def __init__(self, input_d, alpha, fc1=256, fc2=256, ckp_dir="./PPO_scratch/tmp/ppo"):
        super(PPO_criticnet, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_d, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.checkpoint_file = os.path.join(ckp_dir, "critic_torch_ppo")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class PPO_agent:
    def __init__(self, n_actions, input_d, gamma=0.99, alpha=0.0003, 
                 gae_lambda=0.95, policy_clip=0.2, batchsize=64, N=2048, n_epochs=10) -> None:
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = PPO_actornet(n_actions, input_d, alpha)
        self.critic = PPO_criticnet(input_d, alpha)
        self.memory = PPO_memory(batchsize)

    def remember(self, state, action, reward, prob, value, done):
        self.memory.store(state, action, reward, prob, value, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    
    def choose_action(self, obs):
        state = torch.tensor([obs], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample() # 通过采样得到动作，dist 是一个概率分布
        prob = torch.squeeze(dist.log_prob(action)).item() 
        action = torch.squeeze(action).item() # 将动作从tensor转为标量
        value = torch.squeeze(value).item()

        return action, prob, value # 返回动作，概率，价值

    def learn(self):
        for _ in range(self.n_epochs):
            # compute Advantages(theta')
            state_arr, action_arr, reward_arr, \
            old_probs_arr, value_arr, dones_arr, \
            batches = self.memory.generate_batches() # 每次iter都生成一遍是为了保证每次iter的batch都是不同的
            # 解释：reward 同定义， value 是critic网络的评估
            # old_probs_arr 是上一个网络运行并存储的概率，在这次learn中
            # theta 不断更新(iter with _ )，但是和 theta' 相关的东西始终从memory中取
            # 这就形成了 off-policy

            advatages = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount \
                    * (reward_arr[k] + self.gamma * value_arr[k + 1] * (1 - int(dones_arr[k])) - value_arr[k])
                    discount *= self.gamma * self.gae_lambda
                    # 这个公式是GAE的公式，用来计算Advantages(theta')，具体的推导可以参考论文
                advatages[t] = a_t
            advatages = torch.tensor(advatages).to(self.actor.device)

            values = torch.tensor(value_arr).to(self.actor.device)

            # learn
            for batch in batches:
                # batch 是一个list，里面是batch_size个index
                states_ = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)  
                old_probs_ = torch.tensor(old_probs_arr[batch]).to(self.actor.device) # 对数概率
                action_ = torch.tensor(action_arr[batch]).to(self.actor.device)
                new_dist = self.actor(states_)
                new_probs_ = new_dist.log_prob(action_)

                critic_value_ = self.critic(states_)
                critic_value_ = torch.squeeze(critic_value_) # 用来计算critic loss

                ratio = torch.exp(new_probs_ - old_probs_)
                term1 = advatages[batch] * ratio
                term2 = advatages[batch] * torch.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip)
                actor_loss = -torch.min(term1, term2).mean()
                returns = advatages[batch] + values[batch]
                critic_loss = (returns - critic_value_)**2
                critic_loss = critic_loss.mean()

                # 对criticloss的解释：critic value 是一个预测值
                # 而returns 实际上是一个真实值，是通过GAE计算出来的，实际上是所有的reward的加权和
                # 由对应的公式可以确认，所以这里用 returns - critic_value_ 来计算critic loss

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

        self.memory.clear() # 每次learn之后清空memory


                


