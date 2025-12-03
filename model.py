import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, board_size, n_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        layers.append(nn.Linear(board_size * board_size + 1, n_size))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers):
            layers.append(nn.Linear(n_size, n_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(n_size, board_size * board_size))

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x, nm_color):
        x = self.flatten(x)
        nm_color = nm_color.reshape(-1, 1)
        x = torch.cat((x, nm_color), dim=1)
        return self.linear_relu_stack(x)
