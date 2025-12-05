import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, n_size, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(n_size, n_size),
            nn.BatchNorm1d(n_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(n_size, n_size),
            nn.BatchNorm1d(n_size),
            nn.Dropout(dropout)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class NeuralNetwork(nn.Module):
    def __init__(self, board_size, n_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()

        self.input_layer = nn.Sequential(
            nn.Linear(board_size * board_size + 1, n_size),
            nn.ReLU()
        )

        blocks = []
        for _ in range(num_layers):
            blocks.append(ResidualBlock(n_size, dropout))
        
        self.hidden_layers = nn.Sequential(*blocks)

        self.output_layer = nn.Linear(n_size, board_size * board_size)

    def forward(self, x, nm_color):
        x = self.flatten(x)
        nm_color = nm_color.reshape(-1, 1)
        x = torch.cat((x, nm_color), dim=1)
        
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        logits = self.output_layer(x)
        return logits
