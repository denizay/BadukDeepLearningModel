import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class GameDataset(Dataset):
    def __init__(self, pkl_path, device="cuda", prefetch=False):
        print(f"Loading {pkl_path}")
        self.boards = None
        self.label_boards = None
        self.label_colors = None
        self._load_data(pkl_path)

        if prefetch:
            print(f"Moving data to {device}")
            self._prefetch_to_device(device)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.label_boards[idx], self.label_colors[idx]

    def _load_data(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = torch.load(f)
        self.boards = data["boards"]
        self.label_boards = data["label_boards"]
        self.label_colors = data["label_colors"]

    def _prefetch_to_device(self, device):
        self.boards = self.boards.to(device)
        self.label_boards = self.label_boards.to(device)
        self.label_colors = self.label_colors.to(device)
