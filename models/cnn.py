import torch.nn as nn
import torch

class J1J2CNNRegressor1D(nn.Module):
    """
    1D CNN for spin chains.

    Input:  (batch_size, 1, n_spins)
    Output: (batch_size, output_dim)  usually output_dim = 1 (energy)
    """

    def __init__(self, n_spins, depth_1=16, depth_2=32, kernel_size=3):
        super().__init__()
        self.n_spins = n_spins

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=depth_1, kernel_size=kernel_size, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=depth_1, out_channels=depth_2, kernel_size=kernel_size, padding=1),
            nn.ELU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, 32, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(depth_2, depth_2),
            nn.ReLU(),
            nn.Linear(depth_2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 1, n_spins)
        """
        h = self.conv(x)               # (B, 32, n_spins)
        h = self.pool(h).squeeze(-1)   # (B, 32)
        out = self.fc(h)               # (B, output_dim)
        return out