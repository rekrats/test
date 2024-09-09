import os
import numpy as np
import torch
import torch.nn as nn

from SAC_utils import SAC_memory, Valuenetwork, ActorNetwork, CriticNetwork

class SAC_Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):

        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = SAC_memory(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha=alpha, input_d=input_dims, n_actions=n_actions,
                                  name='actor', maxaction=env.action_space.high)
        
        self.critic_1 = CriticNetwork(beta=beta, input_d=input_dims, n_actions=n_actions,
                                      name__='critic_1')
        self.critic_2 = CriticNetwork(beta=beta, input_d=input_dims, n_actions=n_actions,
                                      name__='critic_2')
        
        self.valuenet = Valuenetwork(beta=beta, input_d=input_dims, name='value')
        self.target_valuenet = Valuenetwork(beta=beta, input_d=input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1) # tau=1: copy the weights to target networks

    def choose_action(self, observation):
        obsarray = observation[0]
        state = torch.Tensor(obsarray).to(self.actor.device) # obs to tensor:state
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.replay_buffer.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_valuenet.named_parameters()
        value_params = self.valuenet.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        # update the target value network from the value network
        # target = tau*value + (1-tau)*target
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()

        self.target_valuenet.load_state_dict(value_state_dict)

    def save_models(self):
        print('...saving models...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.valuenet.save_checkpoint()
        self.target_valuenet.save_checkpoint()

    def load_models(self):
        print('...loading models...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.valuenet.load_checkpoint()
        self.target_valuenet.load_checkpoint()

    def learn(self):
        state, action, reward, new_state, done = \
                self.replay_buffer.sample_buffer(self.batch_size)
        state = torch.Tensor(state, dtype=torch.float).to(self.actor.device)
        new_state = torch.Tensor(new_state, dtype=torch.float).to(self.actor.device)
        action = torch.Tensor(action, dtype=torch.float).to(self.actor.device)
        reward = torch.Tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.Tensor(done).to(self.actor.device)

        # compute value
        value = self.valuenet(state).view(-1)
        value_ = self.target_valuenet(new_state).view(-1)
        value_[done] = 0.0
        # 解释：我们需要让 value 尽可能接近一个目标值，这一部分用 V 计算
        # 而 value_ 是用来更新 q1 和 q2 的，用 target V 计算
        # V = E[Q - alpha*log(pi)] 我们需要让 V 尽可能接近这个目标值
        # 用采样消去 E, 利用数据中的 s_t, a_t, 计算出 Q, pi, 完成 V 的更新

        # value loss
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1) # .view(-1) 将确保 log_probs 是一个一维向量
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        value_target = critic_value - log_probs # 这里为了代码的简化，我们将 alpha 设为 1，但是这样效果会变差，因为 alpha 是一个动态的参数
        value_loss = 0.5 * nn.functional.mse_loss(value, value_target)
        self.valuenet.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.valuenet.optimizer.step()

        # actor loss
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        # reparameterize=True 使得我们可以使用梯度下降来更新 actor
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        # 解释：actor_loss = D_KL(π|π_new) - Q(s, π(s))
        # 这个 KL 散度最后可以转化成 E[alpha*log(ou) - Q(s, a)]

        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # critic loss: q1 and q2
        q_target = reward*self.scale + self.gamma*value_ 
        # 这里的 scale 是为了放大 reward 的范围,是一个实践技巧
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        q1_loss = nn.functional.mse_loss(q1_old_policy, q_target)
        q2_loss = nn.functional.mse_loss(q2_old_policy, q_target)
        qtotal_loss = 0.5*q1_loss + 0.5*q2_loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        qtotal_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # soft update target value network
        self.update_network_parameters()

