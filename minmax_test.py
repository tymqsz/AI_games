from minimax import MiniMax
from TicTacToe import TicTacToe

game = TicTacToe(p1_bot=True, p2_bot=True)

model = MiniMax(game=game)

maxi_player = [True, False]
idx = 0
while not game.EOG:
    maxi = maxi_player[idx % 2]

    if idx % 2 == 0 or idx % 2 == 1:
        move = model.get_best_move(game.BOARD, maxi)
        game.set_input(move[0], move[1])

    game.play()
    idx += 1
