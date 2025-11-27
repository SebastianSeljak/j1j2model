import torch.nn as nn

class J1J2ComplexCNNRegressor1D(nn.Module):
    """
    1D Complex-valued CNN for spin chains.

    Input:  (batch_size, 1, n_spins)
    Output: (batch_size, 2) log-amplitude and phase
    """
    
    def __init__(self, n_spins, depth_1=16, depth_2=32, kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, depth_1, kernel_size=kernel_size, padding=1),
            nn.ELU(),
            nn.Conv1d(depth_1, depth_2, kernel_size=kernel_size, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.amplitude_head = nn.Linear(depth_2, 1)
        self.phase_head = nn.Linear(depth_2, 1)

    def forward(self, x):
        features = self.net(x)
        log_amplitude = self.amplitude_head(features)
        phase = self.phase_head(features)
        return log_amplitude, phase