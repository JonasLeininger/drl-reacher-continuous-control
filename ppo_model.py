import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPOModel(nn.Modul):
    
    def __init__(self, state_dim, hidden_units=(64, 64)):
        super(PPOModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], ac)
    
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer