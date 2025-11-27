import torch.nn as nn
import torch

class J1J2CNNRegressor1D(nn.Module):
    """
    1D CNN for spin chains.

    Input:  (batch_size, 3, n_spins)
    Output: (batch_size, output_dim)  usually output_dim = 1 (energy)
    """

    def __init__(self, n_spins: int, output_dim: int = 1):
        super().__init__()
        self.n_spins = n_spins

        # Convolutional feature extractor over the spin chain
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ELU()
        )

        # Pool over the chain length (global average pooling)
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, 32, 1)

        # Fully-connected head
        self.fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 3, n_spins)
        """
        h = self.conv(x)               # (B, 32, n_spins)
        h = self.pool(h).squeeze(-1)   # (B, 32)
        out = self.fc(h)               # (B, output_dim)
        return out