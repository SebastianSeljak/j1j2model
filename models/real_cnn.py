import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class RealCNN(nn.Module):
    """
    Physics-Informed Real-Valued CNN (Your Contribution).
    Outputs log-amplitude only. Relies on Marshall Sign Rule.
    """
    def __init__(self, n_spins, kernel_size=3):
        super().__init__()
        # Feature Extractor
        self.conv1 = nn.Conv1d(1, 16, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size, padding=1)
        # Global Average Pooling (Adaptive to any N)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x).squeeze()
