import torch
from tqdm import tqdm

PATH = "train_data_big.pkl"
OUT = "train_data_big_t2.pkl"

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return torch.load(f)

data = load_data(PATH)
new_data = []

new_data = []

# Assuming 'data' contains tuples: (curr_pos_matrix, label, turn)
# turn should be: 1 for Black, -1 (or 0) for White

for datum in tqdm(data):
    curr_pos, label, turn = datum
    
    # Ensure curr_pos is a float tensor
    curr_pos = curr_pos.float()
    H, W = curr_pos.shape
    
    # Initialize the (3, H, W) tensor
    combined_pos = torch.zeros((3, H, W), dtype=torch.float32, device=curr_pos.device)

    # ---------------------------------------------------------
    # 1. Determine "Current Player" vs "Opponent" values
    # ---------------------------------------------------------
    if turn == 1:
        # It's Black's turn
        me_val = 1       # Black stones are 1
        opp_val = -1     # White stones are -1
        color_plane_val = 1.0 # "1 if black is to play"
    else:
        # It's White's turn
        me_val = -1      # White stones are -1
        opp_val = 1      # Black stones are 1
        color_plane_val = 0.0 # "0 if white is to play"

    # ---------------------------------------------------------
    # 2. Fill the Planes (Relative Perspective)
    # ---------------------------------------------------------
    # Channel 0: "Xt" -> Presence of CURRENT player's stones
    combined_pos[0] = (curr_pos == me_val)

    # Channel 1: "Yt" -> Presence of OPPONENT'S stones
    combined_pos[1] = (curr_pos == opp_val)

    # Channel 2: "C" -> Colour to play (Constant plane)
    combined_pos[2].fill_(color_plane_val)

    new_data.append((combined_pos, label, turn))

with open(OUT, 'wb') as f:
        torch.save(new_data, f)