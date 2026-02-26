import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

K = 4
D_LATENT = 32
STEER_STRENGTH = 0.5 
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "rpc_dataset"
checkpoint_save = "rpc_full_suite"
os.makedirs(checkpoint_save, exist_ok=True)


class RPC_Encoder(nn.module):
    def