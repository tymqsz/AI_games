import tkinter as tk
import numpy as np

class GameState():
    def __init__(self):
        self.EOG = False
        self.SEEK_HUMAN_INPUT = False
        self.n_plays = 0

        self.selected_cell = -1

        self.P1_bot = False
        self.P2_bot = False

        self.BOARD = [9 for _ in range(9)]

        self.player_symbols = [1, 4]

class TTT(tk.Tk):
    def __init__(self, p1_bot=False, p2_bot=False, render=True):
        super().__init__()
        self.resizable(0, 0)
        self.geometry("800x800")
        self.p1_bot = p1_bot
        self.p2_bot = p2_bot


        bg_color = "#3d464a"   
        grid_color = "#1d2124"
        board_color = "#293942"
        board_highlight = "#2f4a5c"
        button_color = "#236370"
        self.configure(bg=bg_color)

        text_font = ("Arial", 24)
        
        
        self.board_canvas = tk.Canvas(master=self, 
                                    width=450, height=450, highlightthickness=0)
        self.board_canvas.place(y=275, x=175)

        # grid creation
        grid_width=8
        self.board_canvas.create_line(0, 150, 450, 150, width=grid_width, fill=grid_color)
        self.board_canvas.create_line(0, 300, 450, 300, width=grid_width, fill=grid_color)
        self.board_canvas.create_line(150, 0, 150, 450, width=grid_width, fill=grid_color)
        self.board_canvas.create_line(300, 0, 300, 450, width=grid_width, fill=grid_color)

        # grid buttons creatinon
        self.pixel = tk.PhotoImage(width=1, height=1)
        self.cells = []
        for i in range(9):
            self.cells.append(tk.Button(master=self.board_canvas, background=board_color, image=self.pixel,
                                        width=120, height=136, text=" ", font=("Arial", 40), highlightthickness=0,
                                        borderwidth=0, compound="c", activebackground=board_highlight,
                                        command=lambda i=i: self.click_cell(i)))
            y = i // 3   
            x = i % 3
            x_offset = 2*x
            y_offset = 2*y

            self.cells[i].place(x=x_offset+x*150, y=y_offset+y*150)
        
        self.reset_button = tk.Button(master=self, text="reset", font=text_font,
                                      width=5, height=2, background=button_color,
                                      activebackground=board_highlight, highlightthickness=0,
                                      borderwidth=0, command=self.reset)
        self.reset_button.place(x=50, y=50)
        
        self.state = GameState()
        self.state.P1_bot = self.p1_bot
        self.state.P2_bot = self.p2_bot


        if not self.state.P1_bot:
            self.state.SEEK_HUMAN_INPUT = True

    def reset(self):
        self.state = GameState()
        self.state.P1_bot = self.p1_bot
        self.state.P2_bot = self.p2_bot

        self.clear_board()

    def clear_board(self):
        for i in range(len(self.cells)):
            self.cells[i].config(text=" ")
    
    def click_cell(self, cell_idx):
        self.state.selected_cell = cell_idx
        self.move(human=True)

    def move(self, human):
        if self.state.EOG:
            print("end of game")
            return
        
        value = self.state.player_symbols[self.state.n_plays % 2]

        self.state.BOARD[self.state.selected_cell] = value
        self.state.n_plays += 1

        # check if end of game
        self.state.EOG = True if self.evaluate_state(self.state.BOARD) is not None else False

        symbol = "X" if value == 4 else "O"
        self.cells[self.state.selected_cell].config(text=symbol)
        
        if human:
            self.state.SEEK_HUMAN_INPUT = False
        else:
            self.state.SEEK_HUMAN_INPUT = True

    # function allowing outside bots to play game
    def set_input(self, cell):
        self.state.selected_cell = cell
    
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
            return 0 # draw 
        
        return winner
    
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