import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPOModel(nn.Module):
    
    def __init__(self, config, hidden_units=(64, 64)):
        super(PPOModel, self).__init__()
        self.fc1 = nn.Linear(config.state_dim, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], config.action_dim)
        self.std = nn.Parameter(torch.zeros(config.action_dim))

        self.to(config.device)

    def forward(self, state, action=None):
        state = torch.tensor(state)
        x = self.fc1(state)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        mean = F.tanh(x)
        dist = torch.distributions.normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)

        return {'action': action,
                'log_pi': log_prob,
                'entropy': entropy,
                'mean': mean}
    
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer