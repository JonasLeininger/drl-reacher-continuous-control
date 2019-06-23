import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):
    
    def __init__(self, config, hidden_units=(64, 64)):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(config.state_dim, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
    
    def forward(self, state):
        x = self.fc1(state)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.relu(x)