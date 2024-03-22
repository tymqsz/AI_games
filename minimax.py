import numpy as np

class MiniMax():
    def __init__(self, game):
        self.game = game

    def get_best_move(self, state, maximize):
        best_move = [-1, -1]

        """ go through all possible moves and select
            the one with the highest/lowest score
            (for maximizing/minimizing player) """
        if maximize:
            best_value = -np.inf
            for move in self.game.possible_moves(state):
                future_state = self.game.future_state(state, move, 0)

                value = self.minimax(future_state, -np.inf, np.inf, maximize=False)

                if value > best_value:
                    best_value = value
                    best_move = move
        else:
            best_value = np.inf
            for move in self.game.possible_moves(state):
                future_state = self.game.future_state(state, move, 1)

                value = self.minimax(future_state, -np.inf, np.inf, maximize=True)
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_move

    def minimax(self, state, alpha, beta, maximize):
        winner = self.game.evaluate_state(state)
        if winner is not None:
            return winner # end of game

        # go through all possible moves
        if maximize:
            max_val = -np.inf
            for move in self.game.possible_moves(state):
                future_state = self.game.future_state(state, move, 0) # calculate next state

                candidate = self.minimax(future_state, alpha, beta, maximize=False) # get value of next state
                max_val = max(max_val, candidate) # update optimal value

                alpha = max(alpha, max_val) # update optimal alternative for maximizer
                
                if beta < alpha:
                    break # alpha-beta pruning

            return max_val
        else:
            min_val = np.inf
            for move in self.game.possible_moves(state):
                future_state = self.game.future_state(state, move, 1)

                candidate = self.minimax(future_state, alpha, beta, maximize=True)
                min_val = min(min_val, candidate)

                beta = min(beta, candidate)

                if beta < alpha:
                    break
 
            return min_val
