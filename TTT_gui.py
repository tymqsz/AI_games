import tkinter as tk
from tkinter import ttk
import numpy as np
from minimax import MiniMax
import threading
from time import sleep

class GameState():
    def __init__(self):
        self.EOG = False
        self.SEEK_HUMAN_INPUT = False
        self.n_plays = 0

        self.selected_cell = -1

        # change for enum
        self.players = ["human", "human"]

        self.BOARD = [9 for _ in range(9)]

        self.player_symbols = [1, 4]

class TTT(tk.Tk):
    def __init__(self):
        super().__init__()
        self.resizable(0, 0)
        self.geometry("800x800")
        self.bot_reload = 0.4

        self.MiniMax = MiniMax(game=self, shuffle_moves=True)

        bg_color = "#3d464a"   
        grid_color = "#1d2124"
        board_color = "#293942"
        board_highlight = "#2f4a5c"
        button_color = "#4c5e6e"
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

        # grid buttons creation
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
        
        self.restart_button = tk.Button(master=self, text="restart", font=text_font,
                                      width=5, height=2, background=button_color,
                                      activebackground=board_highlight, highlightthickness=0,
                                      borderwidth=0, command=self.reset)
        self.restart_button.place(x=50, y=50)

        self.won_label = tk.Label(master=self, text=" ", background=button_color, compound="c",
                                  image=self.pixel, width=250, height=75, font=text_font)
        self.won_label.place(x=275, y=125)

        self.O_label = tk.Label(master=self, font=text_font, text="O: ", background=button_color)
        self.X_label = tk.Label(master=self, font=text_font, text="X: ", background=button_color)
        self.p1_selection = ttk.Combobox(master=self, state="readonly",
                                         values=["human", "minimax"], width=10, height=5, font=text_font)
        self.p2_selection = ttk.Combobox(master=self, state="readonly",
                                         values=["human", "minimax"], width=10, height=5, font=text_font)
        self.p1_selection.place(x=600, y=25)
        self.p2_selection.place(x=600, y=75)
        self.O_label.place(x=550, y=25)
        self.X_label.place(x=550, y=75)
        
        self.p1_selection.bind("<<ComboboxSelected>>", self.change_p1)
        self.p2_selection.bind("<<ComboboxSelected>>", self.change_p2)

        self.state = GameState()
        
        if self.state.players[0] != "human":
            self.state.bot_idx.append(0)
        if self.state.players[1] != "human":
            self.state.bot_idx.append(1)


        if self.state.players[0] == "human":
            self.state.SEEK_HUMAN_INPUT = True

        game_thread = threading.Thread(target=self.play)
        self.after(100, lambda: game_thread.start())

    def play(self):
        maxi_player = [True, False]

        while True:
            maxi = maxi_player[self.state.n_plays % 2]

            if self.state.players[self.state.n_plays%2] != "human":
                sleep(self.bot_reload)
                move = self.MiniMax.get_best_move(self.state.BOARD, maxi)
                self.state.selected_cell = move
                self.move()

    def reset(self):
        self.state = GameState()
        self.state.players[0] = self.p1_selection.get()
        self.state.players[1] = self.p2_selection.get()

        self.won_label.config(text="")
        self.clear_board()

    def clear_board(self):
        for i in range(len(self.cells)):
            self.cells[i].config(text=" ")
    
    def click_cell(self, cell_idx):
        self.state.selected_cell = cell_idx
        self.move()

    def move(self):
        if self.state.EOG:
            return
        
        value = self.state.player_symbols[self.state.n_plays % 2]

        self.state.BOARD[self.state.selected_cell] = value
        self.state.n_plays += 1

        # check if end of game
        self.state.EOG = True if self.evaluate_state(self.state.BOARD) is not None else False

        symbol = "X" if value == 4 else "O"
        self.cells[self.state.selected_cell].config(text=symbol)
        
        if self.state.EOG:
            winner = self.evaluate_state(self.state.BOARD)
            if winner == 0:
                txt = "Draw!"
            elif winner == 1:
                txt = "\'O\' player won"
            else:
                txt = "\'X\' player won"
            self.won_label.config(text=txt)
    
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
    
    def change_p1(self, event):
        value = self.p1_selection.get()
        self.state.P1 = value

    def change_p2(self, event):
        value = self.p2_selection.get()
        self.state.P2 = value