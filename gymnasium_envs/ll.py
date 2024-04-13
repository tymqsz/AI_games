
import gymnasium as gym

import os, sys

train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1', render_mode='human')

# Get the current working directory
current_dir = os.getcwd()

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Append parent directory to sys.path
sys.path.append(parent_dir)

observation, info = test_env.reset(seed=42)
for _ in range(1000):
   action = test_env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = test_env.step(action)

   if terminated or truncated:
      observation, info = test_env.reset()

test_env.close()

def evaluate_model(model, n_games):
    for _ in range(n_games):
        done = False
        observation, _ = test_env.reset()

        while not done:            
            action = model.act(observation)
            observation, reward, done, _, _ = test_env.step(action)
        print(f"{n_games}th game: {reward}")



def train_model(model, n_games, n_tests=1):
    for game in range(n_games):
        done = False
        obs = train_env.reset()
        while not done:
            action = model.learning_act(obs)
            new_obs, reward, done = train_env.step(action)
            model.store_memory(obs, new_obs, action, reward, done)

            model.learn()
            obs = new_obs
        if game % (n_games//n_tests) == 0:
            evaluate_model(model, 10, 0)


from models.dqn import DQN_Agent

model = DQN_Agent(n_observations=4, n_actions=2)
evaluate_model(model, 10)


