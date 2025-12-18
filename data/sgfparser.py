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
    Parse SGF content and return a list of (board, label_board, player_color, is_pass)
    for all valid moves in the game.
    """
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
    raw_moves = sgf_content.split(';')
    moves = []
    for move in raw_moves:
        if move.startswith('B[') or move.startswith('W['):
            moves.append(move)

    game_samples = []

    # FIX: Iterate through ALL moves to capture the very first move
    for i in range(len(moves)):
        current_move_str = moves[i]

        # Determine who is playing THIS move (for the label)
        if current_move_str.startswith('B['):
            label_color = BLACK
        elif current_move_str.startswith('W['):
            label_color = WHITE
        else:
            continue

        # 1. Create the Label (The move we want the network to predict)
        label_board = np.zeros((board_size, board_size), dtype=int)
        is_pass = False
        
        # Check pass for current move
        if "W[]" in current_move_str or "B[]" in current_move_str:
            is_pass = True
        else:
            # Extract coordinates from "B[xy]" -> "xy"
            pos = current_move_str[2:4]
            assert len(pos) == 2
            label_row, label_col = parse_position(pos)
            label_board[label_row, label_col] = 1

        # 2. SAVE STATE BEFORE UPDATING
        # We save the CURRENT board state and the move that is ABOUT to happen
        game_samples.append((board.copy(), label_board, label_color, is_pass))

        # 3. Update Board (Apply the move so it's ready for the next step)
        if not is_pass:
            # We already parsed row/col for the label, reuse or re-parse
            # (Re-parsing safely here in case of weird logic flow)
            pos = current_move_str[2:4]
            row, col = parse_position(pos)
            
            # Place Stone
            board[row, col] = label_color
            
            # Handle Captures (Standard Go Rules)
            opponent_color = -label_color
            
            # Check neighbors for opponent captures
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if board[nr, nc] == opponent_color:
                        if count_liberties(board, nr, nc) == 0:
                            remove_group(board, nr, nc)
            
            # Check self-capture (Suicide rule)
            if count_liberties(board, row, col) == 0:
                remove_group(board, row, col)

    return game_samples