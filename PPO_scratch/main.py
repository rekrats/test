import gymnasium as gym
import numpy as np  
from PPO_torch import PPO_agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    n_games = 300
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    n_actions = env.action_space.n
    input_dims = env.observation_space.shape[0]

    agent = PPO_agent(n_actions=n_actions, input_d=input_dims, 
                      alpha=alpha, n_epochs=n_epochs, batchsize=batch_size)

    figure_file = './PPO_scratch/plots/lunar_lander.png'
    best_score = env.reward_range[0]
    score_history = []
    n_steps = 0
    learn_iters = 0
    avg_score = 0

    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        score = 0

        while ((not done) and (score < 10000)):
            action, prob, val = agent.choose_action(obs)
            obs_, reward, done, __, info = env.step(action)
            n_steps += 1
            agent.remember(obs, action, reward, prob, val, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            score += reward
            obs = obs_
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            print(f'find new models at game {i}, score {score}, avg_score {avg_score}, saving...')

        # print(f'game {i}, score {score}, avg_score {avg_score}, time steps {n_steps}, learn steps {learn_iters}')

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)



