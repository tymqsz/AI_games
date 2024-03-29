from minimax import MiniMax
from TTT_gui import TTT
import threading
from multiprocessing import Process
game = TTT(p1_bot=True, p2_bot=False)

model = MiniMax(game=game, shuffle_moves=True)

def play():
    maxi_player = [True, False]

    idx = 0
    while True:
        if game.state.EOG:
            idx = 0
        while game.state.SEEK_HUMAN_INPUT:
            pass

        maxi = maxi_player[idx % 2]

        if idx % 2 == 0:
            move = model.get_best_move(game.state.BOARD, maxi)
            game.set_input(move)
            game.move(human=False)
        idx += 1

play_thread = threading.Thread(target=lambda:play(10))

def start_play_thread():
    play_thread.start()

play_process = Process(target=play)
game.after(2000, start_play_thread)
game.mainloop()