import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x, temp=1.0):
        h = F.normalize(self.encoder(x), dim=1)
        logits = self.gate(h)
        s = F.gumbel_softmax(logits, tau=temp, hard=False)
        return h, s, logits


class DynamicsMoE(nn.Module):
    def __init__(self, K, dim):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, dim)) 
            for _ in range(K)
        ])

    def forward(self, h, s):
        preds = torch.stack([m(h) for m in self.experts], dim=1)
        return torch.sum(s.unsqueeze(-1) * preds, dim=1)