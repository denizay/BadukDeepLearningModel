import os
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from sgfparser import get_all_moves


SGF_FOLDER_PATH = "../all_games"
BOARD_POS_COUNT = 16


def apply_symmetry(board, label, k):
    """
    Apply one of 8 symmetries to the board and label.
    k: 0-7
    """
    if k >= 4:
        board = np.flip(board, axis=0)
        label = np.flip(label, axis=0)
        k -= 4
    
    if k > 0:
        board = np.rot90(board, k=k)
        label = np.rot90(label, k=k)
        
    return board.copy(), label.copy()


def transform_to_three_planes(board_matrix, turn):
    """
    Transform turn-based board matrix into 3-plane representation.
    turn: 1.0 for Black, -1.0 for White
    """
    H, W = board_matrix.shape
    combined_pos = np.zeros((3, H, W), dtype=np.float32)

    if turn == 1.0:
        # Black's turn
        me_val = 1.0
        opp_val = -1.0
        color_plane_val = 1.0
    else:
        # White's turn
        me_val = -1.0
        opp_val = 1.0
        color_plane_val = 0.0

    # Plane 0: Current player's stones
    combined_pos[0] = (board_matrix == me_val).astype(np.float32)
    
    # Plane 1: Opponent's stones
    combined_pos[1] = (board_matrix == opp_val).astype(np.float32)

    # Plane 2: Color to play
    combined_pos[2] = color_plane_val

    return combined_pos


def get_positions(sgf_paths):
    boards, label_boards, label_colors = [], [], []
    fail_count = 0
    for sgf_path in tqdm(sgf_paths):
        try:
            with open(sgf_path, 'r', encoding='utf-8') as f:
                sgf_data = f.read()
            
            # Get all valid moves/states from the game
            game_samples = get_all_moves(sgf_data)
            
            if not game_samples:
                continue
                
            # Sample distinct moves
            num_samples = len(game_samples)
            if num_samples <= BOARD_POS_COUNT:
                indices = np.arange(num_samples)
            else:
                indices = np.random.choice(num_samples, BOARD_POS_COUNT, replace=False)
            
            for i, idx in enumerate(indices):
                board_matrix, label_board, label_color, is_pass = game_samples[idx]
                
                # Apply symmetries sequentially
                k = i % 8
                board_matrix, label_board = apply_symmetry(board_matrix, label_board, k)
                label_board = np.append(label_board, 1 if is_pass else 0)

                # Transform to 3 planes
                combined_pos = transform_to_three_planes(board_matrix, label_color)
                
                boards.append(torch.tensor(combined_pos, dtype=torch.int8))
                label_boards.append(torch.tensor(label_board, dtype=torch.int8))
                label_colors.append(torch.tensor(label_color, dtype=torch.int8))
                        
        except Exception as e:
            print(f"Error processing {sgf_path}: {e}")
            fail_count += 1
    data = {
        "boards": torch.stack(boards),
        "label_boards": torch.stack(label_boards),
        "label_colors": torch.stack(label_colors)
    }
    return data, fail_count


def main():
    sgf_paths = glob(os.path.join(SGF_FOLDER_PATH, "*.sgf"))
    sgf_count = len(sgf_paths)

    train_paths, val_paths, test_paths = np.split(np.random.permutation(
        sgf_paths), [int(.75 * sgf_count), int(.875 * sgf_count)])

    train_data, train_fc = get_positions(train_paths)
    val_data, val_fc = get_positions(val_paths)
    test_data, test_fc = get_positions(test_paths)

    with open('train_data_big.pkl', 'wb') as f:
        torch.save(train_data, f)

    with open('validation_data_big.pkl', 'wb') as f:
        torch.save(val_data, f)

    with open('test_data_big.pkl', 'wb') as f:
        torch.save(test_data, f)

    print(f"Train data length: {len(train_data)}")
    print(f"Val data length: {len(val_data)}")
    print(f"Test data length: {len(test_data)}")
    fail_count = train_fc + test_fc + val_fc
    print(f"Fail count: {fail_count}")


if __name__ == '__main__':
    main()
