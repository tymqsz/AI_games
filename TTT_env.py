from TTT_gui import TTT
from minimax import MiniMax
import random

class TicTacToe_env():
    def __init__(self, random_move_prob=0.3, DQN_starts=None, draw=False):
        self.random_move_prob = random_move_prob
        self.game = TTT(p1_bot=True, p2_bot=True, render_mode=draw) # initialize game with 2 bots

        # decide whether DQN or MiniMax starts the game
        self.fixed_start = False
        if DQN_starts is not None:
            self.DQN_starts = DQN_starts
            self.fixed_start = True
        else:
            self.DQN_starts = random.choice([True, False])
        
        self.bot_maximizing = False if self.DQN_starts else True


        self.minimax = MiniMax(game=self.game, shuffle_moves=False)

    def reset(self):
        self.game.reset()
        
        if not self.fixed_start:
            self.DQN_starts = random.choice([True, False]) # decide who starts
        self.bot_maximizing = False if self.DQN_starts else True

        # if minimax starts - play first move
        if not self.DQN_starts:
            if random.random() <= self.random_move_prob:
                moves = self.game.possible_moves(self.game.state.BOARD)
                move = random.choice(moves)
            else:
                move = self.minimax.get_best_move(self.game.state.BOARD, maximize=False)

            self.game.set_input(move)
            self.game.next_move()

        observation = self._get_observation()

        return observation
    
    def _get_observation(self):
        observation = self.game.state.BOARD

        return observation

    def step(self, action):
        # check if game is still on
        outcome = self.game.evaluate_state(self.game.state.BOARD)
        if outcome is not None:
            reward = 0 if outcome is None else outcome

            return None, reward, True

        # play DQN's move
        self.game.set_input(action)
        self.game.next_move()
        
        # check if game ended after DQN's move
        observation = self._get_observation()
        outcome = self.game.evaluate_state(self.game.state.BOARD)
        reward = 0 if outcome is None else outcome


        # if not end of game, minimax player makes move
        if outcome is None:
            if random.random() <= self.random_move_prob:
                moves = self.game.get_possible_moves(self.game.state.BOARD)
                move = random.choice(moves)
            else:
                move = self.minimax.get_best_move(self.game.state.BOARD, maximize=self.bot_maximizing)

            self.game.set_input(move)
            self.game.next_move()

            observation = self._get_observation()
            outcome = self.game.evaluate_state(self.game.state.BOARD)
            reward = 0 if outcome is None else outcome

        # if outcome is not None (someone won or draw) - end of game       
        done = True if outcome is not None else False

        return observation, reward, done



