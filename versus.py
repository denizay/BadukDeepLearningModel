import os
import torch
from model import NeuralNetwork
from goban.board import Board
from goban.stone import Stone, BLACK, WHITE


BOARD_SIZE = 9
N_SIZE = 256
NUM_LAYERS = 4
WEIGHTS_PATH = 'checkpoints/mw_256ns_4ls_0.001lr_100ep_256bs.pth'


def play(player_color):
    num_of_moves = 150
    board = Board(BOARD_SIZE)
    board.print_board()

    model = NeuralNetwork(BOARD_SIZE, N_SIZE, NUM_LAYERS)
    model.load_state_dict(
        torch.load(
            WEIGHTS_PATH,
            weights_only=True,
            map_location=torch.device('cpu')))
    model.eval()

    for i in range(num_of_moves):
        if not i % 2:
            color = BLACK
        else:
            color = WHITE

        if color == player_color:
            coords = input("Enter move coordinates:")
            coords = coords.split()
            coords = (int(coords[0]) - 1, ord(coords[1].upper()) - 65)

        else:
            t_board = torch.tensor(board.board, dtype=torch.float)
            t_color = torch.tensor([color], dtype=torch.float)
            t_board = t_board.reshape(1, 9, 9)

            m_move = model(t_board, t_color)

            row, col = divmod(m_move.argmax().item(), 9)
            print(f"row: {row + 1}, col: {col + 1}")
            print(f"confidence: {m_move.max()}")
            coords = (row, col)

        if not coords:
            print("Game Finished")
            break
        x, y = coords
        stone = Stone(color, x, y)
        board.put_stone(stone)
        os.system('cls' if os.name == 'nt' else 'clear')
        board.print_board()


if __name__ == "__main__":
    play(BLACK)
