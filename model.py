import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, board_size, n_filters, num_layers=1, drop_out=0.2, activation=nn.ReLU):
        super().__init__()
        self.board_size = board_size
        self.activation = activation

        # Initial convolutional layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, n_filters, kernel_size=3, padding=1),  # 2 input channels: board state + nm_color
            activation(),
            nn.Dropout(drop_out)
        )

        # Residual convolutional layers
        self.res_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.res_layers.append(
                ResidualConvBlock(n_filters, drop_out, activation)
            )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(n_filters, 1, kernel_size=1),  # Single output channel
            nn.Flatten(),
            nn.Linear(board_size * board_size, board_size * board_size)  # Output logits for each board position
        )

    def forward(self, x, nm_color):
        """
        Args:
            x: Tensor of shape (batch_size, board_size, board_size)
            nm_color: Tensor of shape (batch_size, 1), indicating the player's color
        """
        batch_size = x.shape[0]

        # Add an additional channel for `nm_color`
        nm_color_channel = nm_color.view(batch_size, 1, 1, 1).expand(-1, 1, self.board_size, self.board_size)
        x = torch.cat((x.unsqueeze(1), nm_color_channel), dim=1)  # Shape: (batch_size, 2, board_size, board_size)

        # Apply initial transformation
        x = self.input_layer(x)

        # Apply residual blocks
        for res_layer in self.res_layers:
            x = res_layer(x)

        # Compute output
        x = self.output_layer(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_filters, dropout_rate=0.2, activation=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.Dropout(dropout_rate)
        )
        self.activation = activation()

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity  # Skip connection
        out = self.activation(out)
        return out
