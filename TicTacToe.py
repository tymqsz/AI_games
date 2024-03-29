import numpy as np

class GameState():
    EOG = False
    n_plays = 0

    selected_cell = -1

    P1_bot = False
    P2_bot = False

    BOARD = [0 for _ in range(9)]

    player_symbols = [1, 4]

class TicTacToe():
    def __init__(self, p1_bot=False, p2_bot=False, draw_board=False):
        self.STATE = GameState()
        self.p1_bot = p1_bot
        self.p2_bot = p2_bot

        self.STATE.P1_bot = self.p1_bot
        self.STATE.P2_bot = self.p2_bot

        #self.window = Window(self.BOARD)

    def reset_board(self):
        self.STATE = GameState()   

        self.STATE.P1_bot = self.p1_bot
        self.STATE.P2_bot = self.p2_bot

        self.display_console()
        #self.window.set_state(self.STATE)

    def play(self):
        # update position with appropriate symbol (1/4)
        self.STATE.BOARD[self.STATE.selected_cell] = self.STATE.player_symbols[self.STATE.n_plays % 2]

        self.STATE.n_plays += 1
        
        self.display_console()

        # check if end of game
        self.STATE.EOG = True if self.evaluate_state(self.STATE.BOARD) is not None else False
    
    # function allowing outside bots to play game
    def set_input(self, cell):
        self.STATE.selected_cell = cell
    
    def get_console_input(self):
        row = input("row: ")
        col = input("col: ")
        
        assert(row.isnumeric() and col.isnumeric())

        self.STATE.selected_cell = int(row)*3+int(col)
    
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
        
        if len(TicTacToe.get_possible_moves(BOARD)) == 0 and winner is None:
            return 0 # draw 
        
        return winner
    
    @staticmethod
    def get_possible_moves(BOARD):
        pm = []
        for y in range(3):
            for x in range(3):
                if BOARD[y][x] == 0:
                    pm.append([y, x])

        return pm
    
    # function calculating future state (for minimax alg)
    @staticmethod
    def future_state(BOARD, move, player_idx):
        future_board = BOARD.copy()
        
        future_board[move[0]][move[1]] = TicTacToe.player_symbols[player_idx]

        return future_board

    def display_console(self):
        print("-"*10)
        for y in range(3):
            print("|", end="")
            for x in range(3):
                if self.STATE.BOARD[y*3+x] == 1:
                    symbol = '0'
                elif self.STATE.BOARD[y*3+x] == 4:
                    symbol = 'X'
                else:
                    symbol = ' '

                print(symbol, "|", end="")
            print()
            print("-"*10)
        print("\n")