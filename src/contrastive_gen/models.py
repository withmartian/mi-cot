import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CEBRA_MoE_Encoder", "DynamicsMoE", "SDSSwitch", "SDSRegime"]


class CEBRA_MoE_Encoder(nn.Module):
    def __init__(self, d_in, d_h, K):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, d_h)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_h, 128),
            nn.LayerNorm(128),
            nn.Softplus(),
            nn.Linear(128, K, bias=False)
        )

    def forward(self, x: torch.Tensor, temp: float = 1.0):
        h = F.normalize(self.encoder(x), dim=1)
        logits = self.gate(h)
        s = F.gumbel_softmax(logits, tau=temp, hard=False)
        return h, s, logits


class DynamicsMoE(nn.Module):
    """Mixture-of-experts dynamics model for predicting next latent features."""

    def __init__(self, K: int, dim: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, dim))
                for _ in range(K)
            ]
        )

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        preds = torch.stack([expert(h) for expert in self.experts], dim=1)
        return torch.sum(s.unsqueeze(-1) * preds, dim=1)


class SDSSwitch(nn.Module):
    """Gating network that selects the active dynamical regime."""

    def __init__(self, d: int, K: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LayerNorm(hidden),
            nn.Softplus(),
            nn.Linear(hidden, K, bias=False),
        )

    def forward(self, z: torch.Tensor):
        return self.net(z)


class SDSRegime(nn.Module):
    """A dynamical rule: z_{t+1} = f(z_t)"""
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Softplus(),
            nn.Linear(hidden, d)
        )
    
    def forward(self, z):
        return self.net(z)