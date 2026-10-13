"""Shared MLP + feature-scaling code (used by training, quantization and the app)."""
import numpy as np
import torch
import torch.nn as nn

from src.scaler import Scaler  # re-exported: existing imports keep working


class MLP(nn.Module):
    def __init__(self, in_dim=2381, hidden=(512, 256), p_drop=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict(model, X, device="cpu", bs=4096):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)
