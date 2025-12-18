import os
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from sgfparser import get_all_moves


SGF_FOLDER_PATH = "all_games"

# History Length: How many past moves to include.
# 3 means: Current Board + 3 Previous Boards = 4 Time Steps total.
# Input Depth will be: (HISTORY_LENGTH + 1) * 2 + 1
# For HISTORY_LENGTH=3, Depth = 9 planes.
HISTORY_LENGTH = 3 

def apply_symmetry(input_volume, label, k):
    """
    Apply one of 8 symmetries to the input volume and label.
    input_volume: (Depth, H, W)
    label: (H, W)
    k: 0-7
    """
    # Create copies to avoid mutating original data
    vol = input_volume.copy()
    lbl = label.copy()

    if k >= 4:
        # Flip along Height (axis 1 for volume, axis 0 for label)
        vol = np.flip(vol, axis=1)
        lbl = np.flip(lbl, axis=0)
        k -= 4
    
    if k > 0:
        # Rot90 rotates the first two axes by default.
        # For volume (C, H, W), we want to rotate H, W (axes 1, 2)
        vol = np.rot90(vol, k=k, axes=(1, 2))
        # For label (H, W), we want to rotate axes 0, 1
        lbl = np.rot90(lbl, k=k, axes=(0, 1))
        
    return vol.copy(), lbl.copy()


def encode_history_planes(game_samples, current_idx, history_len):
    """
    Construct input planes using history.
    game_samples: List of (board, label, color, pass)
    current_idx: Index of the current move in game_samples
    """
    # Get the state at the current index
    # We only need the 'color' from the current sample to determine perspective
    _, _, curr_turn_color, _ = game_samples[current_idx]
    
    planes = []
    
    # Iterate from current (0) back to history_len
    for i in range(history_len + 1):
        prev_idx = current_idx - i
        
        # If we go before the start of the game, use an empty board (padding)
        if prev_idx >= 0:
            board_state = game_samples[prev_idx][0]
        else:
            # Assume 9x9 based on current sample or default
            # (Fetching shape from current sample to be safe)
            h, w = game_samples[current_idx][0].shape
            board_state = np.zeros((h, w), dtype=np.float32)

        # Generate 2 planes for this time step:
        # 1. My Stones (relative to CURRENT player)
        # 2. Opponent Stones (relative to CURRENT player)
        
        if curr_turn_color == 1.0: # Current player is Black
            my_stones = (board_state == 1.0)
            opp_stones = (board_state == -1.0)
        else: # Current player is White
            my_stones = (board_state == -1.0)
            opp_stones = (board_state == 1.0)

        planes.append(my_stones.astype(np.float32))
        planes.append(opp_stones.astype(np.float32))

    # Add Color Plane (Last plane)
    # 1.0 if Black to play, 0.0 if White to play
    color_val = 1.0 if curr_turn_color == 1.0 else 0.0
    
    # Use shape from the last processed board
    h, w = planes[0].shape
    color_plane = np.full((h, w), color_val, dtype=np.float32)
    planes.append(color_plane)

    # Stack all planes: Shape (Depth, H, W)
    return np.stack(planes)


def get_positions(sgf_paths):
    boards, label_boards, label_colors = [], [], []
    fail_count = 0
    
    for sgf_path in tqdm(sgf_paths):
        try:
            with open(sgf_path, 'r', encoding='utf-8') as f:
                sgf_data = f.read()
            
            game_samples = get_all_moves(sgf_data)
            
            if not game_samples:
                continue
                
            # Iterate through all moves in the game
            for i in range(len(game_samples)):
                # Extract components from current sample
                # Note: We don't use 'original_board' directly here anymore
                # because 'encode_history_planes' handles retrieval
                _, original_label, label_color, is_pass = game_samples[i]
                
                # Create the stacked input volume with history
                input_volume = encode_history_planes(game_samples, i, HISTORY_LENGTH)
                
                # Apply ALL 8 symmetries
                for k in range(8):
                    sym_vol, sym_label = apply_symmetry(input_volume, original_label, k)
                    
                    # Append pass indicator to label
                    # Label shape becomes (H*W + 1) effectively when flattened, 
                    # but here we keep (H, W) plus separate pass flag later or append now.
                    # Your previous code appended to flattened or kept separate?
                    # Previous code: np.append(label_board, 1 if is_pass else 0)
                    # We will flatten label here for consistency with that logic
                    
                    flat_label = sym_label.flatten()
                    flat_label = np.append(flat_label, 1 if is_pass else 0)

                    boards.append(torch.tensor(sym_vol, dtype=torch.float32)) # Float for CNN inputs
                    label_boards.append(torch.tensor(flat_label, dtype=torch.float32))
                    label_colors.append(torch.tensor(label_color, dtype=torch.float32))
                        
        except Exception as e:
            print(f"Error processing {sgf_path}: {e}")
            fail_count += 1
            
    if not boards:
        return {"boards": [], "label_boards": [], "label_colors": []}, fail_count

    data = {
        "boards": torch.stack(boards),
        "label_boards": torch.stack(label_boards),
        "label_colors": torch.stack(label_colors)
    }
    return data, fail_count


def main():
    if not os.path.exists(SGF_FOLDER_PATH):
        print(f"Folder {SGF_FOLDER_PATH} not found.")
        return

    sgf_paths = glob(os.path.join(SGF_FOLDER_PATH, "*.sgf"))[:5000]
    sgf_count = len(sgf_paths)
    
    if sgf_count == 0:
        print("No SGF files found.")
        return

    # Shuffle and Split
    shuffled_paths = np.random.permutation(sgf_paths)
    train_paths, val_paths, test_paths = np.split(shuffled_paths, 
                                                  [int(.90 * sgf_count), int(.95 * sgf_count)])

    print("Processing Training Data...")
    train_data, train_fc = get_positions(train_paths)
    
    print("Processing Validation Data...")
    val_data, val_fc = get_positions(val_paths)
    
    print("Processing Test Data...")
    test_data, test_fc = get_positions(test_paths)

    # Save
    with open('train_data_history.pkl', 'wb') as f:
        torch.save(train_data, f)

    with open('validation_data_history.pkl', 'wb') as f:
        torch.save(val_data, f)

    with open('test_data_history.pkl', 'wb') as f:
        torch.save(test_data, f)

    if len(train_data["boards"]) > 0:
        print(f"Input Shape: {train_data['boards'][0].shape}") # Should be (9, 9, 9) or similar
        print(f"Train samples: {len(train_data['boards'])}")
        print(f"Val samples: {len(val_data['boards'])}")
        print(f"Test samples: {len(test_data['boards'])}")
    
    fail_count = train_fc + test_fc + val_fc
    print(f"Total failures: {fail_count}")


if __name__ == '__main__':
    main()