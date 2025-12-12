import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import your modules
from dataset import GameDataset
from model import NeuralNetwork

# --- Constants (Matching your loading style) ---
BOARD_SIZE = 9
# You can change these defaults or pass them as arguments
DEFAULT_N_SIZE = 64
DEFAULT_NUM_LAYERS = 12
BATCH_SIZE = 1024
DEFAULT_PATH = "checkpoints"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

DEFAULT_DATA_PATH = "data/test_data_big_t2.pkl"
DEFAULT_MODEL_PATH ='checkpoints/20251212_001130/epoch_121.pth'

def evaluate(data_path, checkpoint_path, n_size, num_layers):
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print(f"Loading dataset: {data_path}")
    # assuming GameDataset handles device placement internally based on your first script
    dataset = GameDataset(data_path, DEVICE, prefetch=False) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    # Using the specific architecture constants you provided
    print(f"Initializing NeuralNetwork(board_size={BOARD_SIZE}, n_size={n_size}, num_layers={num_layers})...")
    model = NeuralNetwork(BOARD_SIZE, n_size, num_layers).to(DEVICE)

    # 3. Load Weights
    print(f"Loading weights from: {checkpoint_path}")
    try:
        # Using the specific loading syntax you requested
        state_dict = torch.load(
            checkpoint_path, 
            map_location=DEVICE, 
            weights_only=True
        )
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    # 4. Evaluation Loop
    total_loss = 0
    correct_top1 = 0
    correct_top3 = 0
    total_samples = len(dataset)
    num_batches = len(dataloader)

    print(f"Starting evaluation on {total_samples} samples...")
    
    with torch.no_grad():
        for X, y, nm_color in dataloader:
            # Ensure data is on the correct device
            # (If GameDataset doesn't put them on device, uncomment below)
            X, y, nm_color = X.to(DEVICE), y.to(DEVICE), nm_color.to(DEVICE)

            pred = model(X)
            
            # Reshape y to (Batch_Size, 81) as per your training logic
            y_reshaped = torch.reshape(y, (-1, BOARD_SIZE * BOARD_SIZE))
            
            # --- Loss ---
            loss = loss_fn(pred, y_reshaped)
            total_loss += loss.item()

            # --- Top-1 Accuracy ---
            target_indices = y_reshaped.argmax(1)
            correct_top1 += (pred.argmax(1) == target_indices).sum().item()

            # --- Top-3 Accuracy ---
            _, top3_pred = pred.topk(3, 1, True, True)
            correct_top3 += top3_pred.eq(target_indices.view(-1, 1)).sum().item()

    # 5. Calculate Metrics
    avg_loss = total_loss / num_batches
    acc_top1 = 100 * correct_top1 / total_samples
    acc_top3 = 100 * correct_top3 / total_samples

    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Loss:           {avg_loss:.6f}")
    print(f"Top-1 Accuracy: {acc_top1:.2f}%")
    print(f"Top-3 Accuracy: {acc_top3:.2f}%")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to the .pkl dataset file")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the .pth checkpoint file")
    parser.add_argument("--n_size", type=int, default=DEFAULT_N_SIZE, help=f"Model N_SIZE (default: {DEFAULT_N_SIZE})")
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS, help=f"Model NUM_LAYERS (default: {DEFAULT_NUM_LAYERS})")

    args = parser.parse_args()

    evaluate(args.data_path, args.checkpoint_path, args.n_size, args.num_layers)