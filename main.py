from environment import PongEnv
from DQN import Agent
from torch import nn 
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

env = gym.make("CartPole-v1")
agent = Agent(env=env, n_actions=2, n_observations=4, lr=1e-4, batch_size=64, memory_cap=10000, gamma=0.99)


# TRAINING
n_episodes = 500

plot_data = deque(maxlen=n_episodes)
fig, ax = plt.subplots() 
line = ax.plot([])[0]
ax.set_xlim(0, n_episodes) 
ax.set_ylim(0, 500) 

episode = 0
while episode < n_episodes:
    done = False
    obs, _ = env.reset()
    
    reward_sum = 0
    while not done:

        action = int(agent.act(obs))
        new_obs, reward, done, _, _ = env.step(action)
        
        agent.store_memory(obs, new_obs, action, reward, done)

        reward_sum += reward
        obs = new_obs

        agent.learn()
    
    plot_data.append((episode, reward_sum)) 
  
    x_values = [x for x, _ in plot_data] 
    y_values = [y for _, y in plot_data] 
    line.set_data(x_values, y_values) 
    plt.pause(0.01)
    episode += 1

    if reward_sum >= 500:
        print("reached 500 :)")
        break

env = gym.make("CartPole-v1", render_mode="human")
# TESTING
n_episodes = 5
episode = 0
while episode < n_episodes:
    done = False
    obs, _ = env.reset()
    
    reward_sum = 0
    while not done:
        action = int(agent.act(obs))
        new_obs, reward, done, _, _ = env.step(action)

        reward_sum += reward
        obs = new_obs
    
    print(f"{episode}th reward: {reward_sum}")
    episode += 1