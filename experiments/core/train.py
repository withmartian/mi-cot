import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import numpy as np

from contrastive_gen.models import CEBRA_MoE_Encoder, DynamicsMoE
from contrastive_gen.losses import nce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_eval_k(K_val, features, X_torch, triplets, epochs=50):
    print(f"\n--- Evaluating K={K_val} ---", flush=True)
    d_h = 32
    model = CEBRA_MoE_Encoder(X_torch.shape[1], d_h, K_val).to(device)
    dyn = DynamicsMoE(K_val, d_h).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

    for epoch in range(epochs):
        tau = max(0.2, 1.5 * (0.92 ** epoch))
        curr_w_div = min(30.0, (epoch / 15.0) * 30.0)
        
        indices = np.random.permutation(len(triplets))
        for b in range(0, len(triplets), 128):
            b_idx = indices[b:b+128]
            i_t = torch.tensor([triplets[x][0] for x in b_idx], device=device)
            p_t = torch.tensor([triplets[x][1] for x in b_idx], device=device)
            n_t = torch.tensor([triplets[x][2] for x in b_idx], device=device)

            h_i, s_i, _ = model(X_torch[i_t], temp=tau)
            h_p, s_p, _ = model(X_torch[p_t], temp=tau)
            h_n, _, _ = model(X_torch[n_t], temp=tau)
            
            h_pred = dyn(h_i, s_i)
            
            l_nce = nce_loss(h_pred, h_p, h_n)
            l_mse = F.mse_loss(h_pred, h_p)
            l_div = (s_i.mean(0) * torch.log(s_i.mean(0) + 1e-8)).sum()
            l_pers = torch.abs(s_i - s_p).mean()

            loss = l_nce + (10.0 * l_mse) + (curr_w_div * l_div) + (10.0 * l_pers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, s_final, _ = model(X_torch, temp=0.01)
        states = s_final.argmax(1).cpu().numpy()
    
    pids = [f['problem_id'] for f in features]
    p_scores = []
    p_map = defaultdict(list)
    for i, p in enumerate(pids):
        p_map[p].append(states[i])
    for p, seq in p_map.items():
        if len(seq) > 1:
            p_scores.append(1.0 - (np.sum(np.array(seq[1:]) != np.array(seq[:-1])) / (len(seq)-1)))
    
    return np.mean(p_scores), l_mse.item(), model, dyn, states