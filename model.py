import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, num_planes, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_planes, num_planes, 3, padding=1),
            nn.BatchNorm2d(num_planes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(num_planes, num_planes, 3, padding=1),
            nn.BatchNorm2d(num_planes),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class NeuralNetwork(nn.Module):
    def __init__(self, board_size, num_planes=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()

        self.input_layer = nn.Sequential(
            # blacks, whites, turn, 3 features for input
            nn.Conv2d(3, num_planes, 3, padding=1),
            nn.BatchNorm2d(num_planes),
            nn.ReLU(),
        )

        blocks = []
        for _ in range(num_layers):
            blocks.append(ResidualBlock(num_planes, dropout))

        self.hidden_layers = nn.Sequential(*blocks)

        self.policy_reduce = nn.Sequential(
            nn.Conv2d(num_planes, 2, 1), nn.BatchNorm2d(2), nn.ReLU()
        )

        self.output_layer = nn.Linear(
            board_size * board_size * 2, board_size * board_size
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.policy_reduce(x)
        x = self.flatten(x)
        logits = self.output_layer(x)
        return logits
