#from TicTacToe import TTT
import random
import os, sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.minimax import MiniMax

class GameState():
    def __init__(self):
        self.EOG = False
        self.n_plays = 0

        self.selected_cell = -1

        # change for enum
        self.players = ["bot", "bot"]

        self.BOARD = [9 for _ in range(9)]

        self.player_symbols = [1, 4]

class TTT():
    def __init__(self):
        self.state = GameState()

    def reset(self):
        self.state = GameState()

        self.state.players[0] = "bot"
        self.state.players[1] = "bot"

    def move(self):
        if self.state.EOG:
            return
        
        value = self.state.player_symbols[self.state.n_plays % 2]

        self.state.BOARD[self.state.selected_cell] = value
        self.state.n_plays += 1

        # check if end of game
        self.state.EOG = True if self.evaluate_state(self.state.BOARD) is not None else False
    
    @staticmethod
    def evaluate_state(BOARD):
        winner = None # non ending state

        # change format for easier calc
        BOARD = np.array(BOARD).reshape(3, 3)

        # check rows and columns
        for i in range(3):
            if BOARD[i].sum() == 3 or BOARD[:, i].sum() == 3:
                winner = 1
            if BOARD[i].sum() == 12 or BOARD[:, i].sum() == 12:
                winner = -1

        # check diagonals
        diag1_sum, diag2_sum = 0, 0
        for i in range(3):
            diag1_sum += BOARD[i, i]
            diag2_sum += BOARD[2-i, i]
        
        if diag1_sum == 3 or diag2_sum == 3:
            winner = 1
        if diag1_sum == 12 or diag2_sum == 12:
            winner = -1
        
        if len(TTT.get_possible_moves(BOARD.flatten())) == 0 and winner is None:
            return 0.95 # draw 
        
        return winner


    def display_console(self):
        print("-"*10)
        for y in range(3):
            print("|", end="")
            for x in range(3):
                if self.state.BOARD[y*3+x] == 1:
                    symbol = '0'
                elif self.state.BOARD[y*3+x] == 4:
                    symbol = 'X'
                else:
                    symbol = ' '

                print(symbol, "|", end="")
            print()
            print("-"*10)
        print("\n")
    
    @staticmethod
    def get_possible_moves(BOARD):
        pm = []
        for move in range(9):
            if BOARD[move] == 9:
                pm.append(move)

        return pm
    
    # function calculating future state (for minimax alg)
    def future_state(self, BOARD, move, player_idx):
        future_board = BOARD.copy()
        
        future_board[move] = self.state.player_symbols[player_idx]

        return future_board

class TicTacToe_env():
    def __init__(self, random_move_prob=0.3, DQN_starts=True, draw=False):
        self.draw = draw
        self.random_move_prob = random_move_prob
        self.game = TTT() # initialize game with 2 bots

        # decide whether DQN or MiniMax starts the game
        self.DQN_starts = DQN_starts
        
        self.bot_maximizing = False if self.DQN_starts else True

        self.minimax = MiniMax(game=self.game, shuffle_moves=True)

    def reset(self):
        self.game.reset()

        self.bot_maximizing = False if self.DQN_starts else True

        # if minimax starts - play first move
        if not self.DQN_starts:
            if random.random() <= self.random_move_prob:
                moves = self.game.get_possible_moves(self.game.state.BOARD)
                move = random.choice(moves)
            else:
                move = self.minimax.get_best_move(self.game.state.BOARD, maximize=False)

            self.game.state.selected_cell = move
            self.game.move()

        observation = self._get_observation()

        if self.draw:
            self.game.display_console()
        return observation
    
    def _get_observation(self):
        observation = self.game.state.BOARD
        obs = np.array(observation)
        obs[obs == 1] = -1
        obs[obs == 4] = 1
        obs[obs == 9] = 2
        observation = obs.tolist()
        return observation

    def step(self, action):
        # check if game is still on
        outcome = self.game.evaluate_state(self.game.state.BOARD)
        if outcome is not None:
            reward = 0 if outcome is None else outcome

            return None, reward, True

        # play DQN's move
        self.game.state.selected_cell = action
        self.game.move()
        
        # check if game ended after DQN's move
        observation = self._get_observation()
        outcome = self.game.evaluate_state(self.game.state.BOARD)
        reward = 0 if outcome is None else outcome

        if self.draw:
            self.game.display_console()
        # if not end of game, minimax player makes move
        if outcome is None:
            if random.random() <= self.random_move_prob:
                moves = self.game.get_possible_moves(self.game.state.BOARD)
                move = random.choice(moves)
            else:
                move = self.minimax.get_best_move(self.game.state.BOARD, maximize=self.bot_maximizing)

            self.game.state.selected_cell = move
            self.game.move()

            observation = self._get_observation()
            outcome = self.game.evaluate_state(self.game.state.BOARD)
            reward = 0 if outcome is None else outcome
            if self.draw:
                self.game.display_console()

        # if outcome is not None (someone won or draw) - end of game       
        done = True if outcome is not None else False

        return observation, reward, done