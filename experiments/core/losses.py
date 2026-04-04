import torch
import torch.nn.functional as F

def nce_loss(z, p, n, temp=0.05):
    pos = torch.sum(z * p, dim=1, keepdim=True)
    neg = torch.sum(z * n, dim=1, keepdim=True)
    logits = torch.cat([pos, neg], dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(len(z), dtype=torch.long, device=z.device))