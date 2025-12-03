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


def get_all_moves(sgf_content):
    """
    Parse SGF content and return a list of (board, label_board, player_color)
    for all valid moves in the game.
    """
    # Parse SGF file to find board size and moves
    board_size = 9
    if 'SZ[' in sgf_content:
        try:
            board_size_str = sgf_content.split('SZ[')[1].split(']')[0]
            board_size = int(board_size_str)
        except IndexError:
            pass

    # Initialize the board matrix (0 = empty, 1 = black, -1 = white)
    board = np.zeros((board_size, board_size), dtype=float)

    # Parse moves, assume format ;B[dd];W[pp]...
    # Split at each move
    raw_moves = sgf_content.split(';')
    
    # Filter valid moves
    moves = []
    for move in raw_moves:
        if move.startswith('B[') or move.startswith('W['):
            moves.append(move)

    game_samples = []

    # Iterate through moves to generate states
    # We need at least one move to have a label
    for i in range(len(moves) - 1):
        current_move = moves[i]
        next_move = moves[i+1]

        # 1. Apply current_move to board
        if current_move.startswith('B['):
            color = BLACK
        elif current_move.startswith('W['):
            color = WHITE
        else:
            continue

        pos = current_move[2:4]
        if len(pos) == 2:  # Valid position
            row, col = parse_position(pos)
            board[row, col] = color
            
            # Handle captures
            opponent_color = -color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if board[nr, nc] == opponent_color:
                        if count_liberties(board, nr, nc) == 0:
                            remove_group(board, nr, nc)
            
            # Self-capture check
            if count_liberties(board, row, col) == 0:
                remove_group(board, row, col)

        # 2. Prepare label from next_move
        if next_move.startswith('B['):
            label_color = BLACK
        elif next_move.startswith('W['):
            label_color = WHITE
        else:
            continue # Skip if next move is invalid

        # Skip passes for label generation if desired, or handle them.
        # The original code skipped passes for labels: "not check_pass(label_move)"
        if len(next_move[2:4]) == 2 and not check_pass(next_move):
            label_row, label_col = parse_position(next_move[2:4])
            
            # Create label board (all zeros except target)
            label_board = np.zeros((board_size, board_size), dtype=int)
            label_board[label_row, label_col] = 1
            
            # Append copy of current board, label, and color
            # Note: We append the board state AFTER the current move is played, 
            # which is the state used to predict the NEXT move.
            game_samples.append((board.copy(), label_board, label_color))

    return game_samples
