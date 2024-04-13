from TTT_train_env import TicTacToe_env
import os, sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.dqn import DQN_Agent


def evaluate_model(n_games, rmp=0):
    env = TicTacToe_env(random_move_prob=rmp, DQN_starts=True, draw=False)
    win, loss, draw = 0, 0, 0

    for _ in range(n_games):
        done = False
        observation = env.reset()
        desired_reward = 1 if env.DQN_starts else -1

        while not done:
            lm_mask = [True if x == 2 else False for x in observation]
            
            action = model.act(observation, lm_mask)
            observation, reward, done = env.step(action)
        if reward == desired_reward:
            win += 1
        elif reward == -desired_reward:
            loss += 1
        else:
            draw += 1
    print(f"wins: {win}, losses: {loss}, draws: {draw}")


def train_model(n_games, n_tests=1):
    for game in range(n_games):
        done = False
        obs = train_env.reset()
        while not done:
            lm_mask = [True if x == 2 else False for x in obs]
            action = model.learning_act(obs, lm_mask)
            new_obs, reward, done = train_env.step(action)
            model.store_memory(obs, new_obs, action, reward, done)

            model.learn()
            obs = new_obs
        if game % (n_games//n_tests) == 0:
            evaluate_model(10, 0)

train_env = TicTacToe_env(random_move_prob=0.3, DQN_starts=True, draw=False)

model = DQN_Agent(env=train_env, n_actions=9, n_observations=9, batch_size=64,
                     lr=1e-4, eps_decay=0.995, double_network=True, tau=0.4)

train_model(1000, n_tests=10)
torch.save(model.policy_network.state_dict(), "../weights/O_DDQN.pt")
evaluate_model(10, rmp=0)