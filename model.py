import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, board_size, n_size, num_layers=1, drop_out=0.2):
        super().__init__()
        self.flatten = nn.Flatten()

        # Initial layer
        self.input_layer = nn.Sequential(
            nn.Linear(board_size * board_size + 1, n_size),
            nn.ReLU()
        )

        # Middle layers with residual connections
        self.res_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.res_layers.append(
                ResidualBlock(n_size, drop_out)
            )

        # Output layer
        self.output_layer = nn.Linear(n_size, board_size * board_size)

    def forward(self, x, nm_color):
        x = x.float()
        x = self.flatten(x)
        nm_color = nm_color.reshape(-1, 1).float()
        x = torch.cat((x, nm_color), dim=1)

        # Initial transformation
        x = self.input_layer(x)

        # Residual blocks
        for res_layer in self.res_layers:
            x = res_layer(x)

        # Final output
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, n_size, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(n_size, n_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_size, n_size),
            nn.Dropout(dropout_rate)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity  # Skip connection
        out = self.relu(out)
        return out
