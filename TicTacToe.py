import tkinter
import numpy as np

class Window():
    symbols = [' ', 'O', ' ', ' ', 'X'] # 1st indx - O, 4th - X
    
    def __init__(self, board):
        self.board = board

    def display_console(self):
        print("-"*10)
        for y in range(3):
            print("|", end="")
            for x in range(3):
                symbol = Window.symbols[int(self.board[y][x])]
                print(symbol, "|", end="")
            print()
            print("-"*10)
        print("\n")

class TicTacToe():
    player_symbols = [1, 4]
    def __init__(self, p1_bot=False, p2_bot=False):
        self.BOARD = np.zeros((3, 3))
        self.P1_bot = p1_bot
        self.P2_bot = p2_bot

        self.window = Window(self.BOARD)

        self.n_plays = 0
    
        self.selected_row = -1
        self.selected_col = -1

        self.EOG = False

    def play(self):
        winner = self.evaluate_state(self.BOARD)
        if winner is not None:
            if winner == 0:
                print("Draw")
            else:
                print(f"Winner: P{winner}")

            self.EOG = True
            self.display()
            return
        
        if self.n_plays % 2 == 0:
            if not self.P1_bot:
                self.get_console_input()
        else:
            if not self.P2_bot:
                self.get_console_input()

        if self.BOARD[self.selected_row][self.selected_col] != 0:
            print(f"invalid play\nWinner: P{(self.n_plays+1) % 2 + 1}")
            
            self.EOG = True
            return
            
        self.BOARD[self.selected_row][self.selected_col] = TicTacToe.player_symbols[self.n_plays % 2]

        self.n_plays += 1
        self.display()
    
    def set_input(self, row, col):
        self.selected_row = int(row)
        self.selected_col = int(col)
    
    def get_console_input(self):
        row = input("row: ")
        col = input("col: ")
        
        assert(row.isnumeric() and col.isnumeric())

        self.selected_row = int(row)
        self.selected_col = int(col)
    
    @staticmethod
    def evaluate_state(BOARD):
        winner = None # non ending state

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
        

        if len(TicTacToe.possible_moves(BOARD)) == 0 and winner is None:
            return 0 # draw 
        
        return winner
    
    @staticmethod
    def possible_moves(BOARD):
        pm = []
        for y in range(3):
            for x in range(3):
                if BOARD[y][x] == 0:
                    pm.append([y, x])

        return pm
    
    @staticmethod
    def future_state(BOARD, move, player_idx):
        future_board = BOARD.copy()
        
        future_board[move[0]][move[1]] = TicTacToe.player_symbols[player_idx]

        return future_board
    
    def display(self):
        self.window.display_console()