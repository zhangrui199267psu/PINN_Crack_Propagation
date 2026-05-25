import torch
import torch.nn as nn


class CrackPINN(nn.Module):
    """Minimal PINN for 2D Mode-I crack-like displacement field u=(u_x,u_y)."""

    def __init__(self, width=32, depth=3):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
