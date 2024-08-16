import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class GameDataset(Dataset):
    def __init__(self, pkl_path, device="cuda", prefetch=False):
        print(f"Loading {pkl_path}")
        self.data = self._load_data(pkl_path)

        if prefetch:
            print(f"Moving data to {device}")
            self._prefetch_to_device(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _load_data(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            return torch.load(f)

    def _prefetch_to_device(self, device):
        self.data = [
            (board.to(device), label.to(device), label_color.to(device))
            for board, label, label_color in tqdm(self.data)
        ]
