from TTT_train_env import TicTacToe_env
from TicTacToe import TTT
import numpy as np
import torch
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.dqn import DQN_Agent


env = TicTacToe_env(random_move_prob=0, DQN_starts=True, draw=False)

model = DQN_Agent(env=env, n_actions=9, n_observations=9, batch_size=64,
              lr=1e-3, eps_decay=0.999, double_network=True, tau=0.4)

def evaluate_model(n_games):
    test_env = TicTacToe_env(random_move_prob=0, DQN_starts=True, draw=False)
    win, loss, draw = 0, 0, 0

    for _ in range(n_games):
        done = False
        observation = test_env.reset()
        desired_reward = 1 if test_env.DQN_starts else -1

        while not done:
            lm_mask = [True if x == 9 else False for x in observation]
            
            action = model.act(observation, lm_mask)

            observation, reward, done = test_env.step(action)
        
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
        obs = env.reset()
        while not done:
            lm_mask = [True if x == 9 else False for x in obs]
            action = model.learning_act(obs, lm_mask)

            new_obs, reward, done = env.step(action)
            model.store_memory(obs, new_obs, action, reward, done)

            model.learn()
            obs = new_obs
        if game % (n_games//n_tests) == 0:
            evaluate_model(3)

        
def play_against(n_games):
    game = TTT(p1_bot=True, p2_bot=False, render_mode="ok")

    for _ in range(n_games):
        i = 0
        game.reset()
        while not game.state.EOG:
            if i % 2 == 0:
                lm_mask = [True if x == 9 else False for x in np.array(game.state.BOARD)]
                action = model.act(game.state.BOARD, lm_mask)

                print("act", action)
                game.set_input(action)
            game.next_move()
            i += 1


train_model(800, n_tests=8)
torch.save(model.policy_network.state_dict(), "models/TTT_O_DDQN_agent.pt")
#evaluate_model(10)
#play_against(2)