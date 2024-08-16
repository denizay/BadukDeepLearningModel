import random
import numpy as np


BLACK = 1.0
WHITE = -1.0

PASSC = 0


def check_pass(pos):
    global PASSC
    if "W[]" in pos or "B[]" in pos:
        PASSC += 1
        return True
    return False


def parse_position(pos):
    """Parse the position from SGF format to board indices."""
    col = ord(pos[0]) - ord('a')
    row = ord(pos[1]) - ord('a')
    return row, col


def count_liberties(board, row, col):
    """Count liberties of a stone at (row, col)."""
    color = board[row, col]
    visited = set()
    stack = [(row, col)]
    liberties = 0

    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        # Check all four directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if board[nr, nc] == 0:
                    liberties += 1
                elif board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))

    return liberties


def remove_group(board, row, col):
    """Remove the group of stones connected to (row, col)."""
    color = board[row, col]
    stack = [(row, col)]
    group = []

    while stack:
        r, c = stack.pop()
        if (r, c) in group:
            continue
        group.append((r, c))

        # Check all four directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if board[nr, nc] == color and (nr, nc) not in group:
                    stack.append((nr, nc))

    for r, c in group:
        board[r, c] = 0


def sgf_to_matrix(sgf_content):
    # Parse SGF file to find board size and moves
    board_size = 9
    if 'SZ[' in sgf_content:
        board_size_str = sgf_content.split('SZ[')[1].split(']')[0]
        board_size = int(board_size_str)

    # Initialize the board matrix (0 = empty, 1 = black, 2 = white)
    board = np.zeros((board_size, board_size), dtype=float)

    # Parse moves, assume format ;B[dd];W[pp]...
    moves = sgf_content.split(';')[1:]  # Split at each move

    moves = [move for move in moves if move.startswith(
        'B[') or move.startswith('W[')]

    bpos_move = random.randint(0, len(moves) - 2)

    label_move = moves[bpos_move + 1]
    if label_move.startswith('B['):
        label_color = BLACK
    elif label_move.startswith('W['):
        label_color = WHITE
    else:
        return None

    label_board = np.zeros((board_size, board_size), dtype=int)
    # print("label move:")
    # print(label_move)

    if len(label_move[2:4]) == 2 and not check_pass(label_move):
        # print((label_move[2:4]))
        # print(len(label_move[2:4]))
        label_row, label_col = parse_position(label_move[2:4])
        # label_board[label_row, label_col] = label_color
        label_board[label_row, label_col] = 1

    for move in moves[:bpos_move + 1]:
        if move.startswith('B['):
            color = BLACK
        elif move.startswith('W['):
            color = WHITE
        else:
            continue

        pos = move[2:4]
        # print(move)
        if len(pos) == 2:  # Valid position
            row, col = parse_position(pos)
            board[row, col] = color

            # Check for captures around the placed stone
            # opponent_color = 3 - color
            opponent_color = - color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if board[nr, nc] == opponent_color:
                        if count_liberties(board, nr, nc) == 0:
                            remove_group(board, nr, nc)

            # Check if the placed stone has no liberties (self-capture)
            if count_liberties(board, row, col) == 0:
                remove_group(board, row, col)

    return board, label_board, label_color
