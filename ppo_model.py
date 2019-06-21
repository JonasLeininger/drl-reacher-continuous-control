import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base_model import BaseModel

class PPOModel(nn.Module):
    
    def __init__(self, config, actor_body, critic_body, hidden_units=(64, 64)):
        super(PPOModel, self).__init__()
        self.actor = actor_body
        self.critic = critic_body
        self.actor_action = nn.Linear(hidden_units[0], config.action_dim)
        self.critic_value = nn.Linear(hidden_units[1], 1)
        self.std = nn.Parameter(torch.zeros(config.action_dim))
        self.device = config.device
        self.to(self.device)

    def forward(self, state, action=None):
        phiA = self.actor(state)
        x = self.actor_action(phiA)
        mean = torch.tanh(x)

        phiV = self.critic(state)
        value = self.critic_value(phiV)

        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)

        return {'actions': action,
                'values' : value,
                'log_pi': log_prob,
                'entropy': entropy,
                'mean': mean}
    
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer