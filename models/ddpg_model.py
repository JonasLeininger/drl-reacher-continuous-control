import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGModel(nn.Module):

    def __init__(self, config, actor_body, critic_body, hidden_units=(300, 300)):
        super(DDPGModel, self).__init__()
        self.actor = actor_body
        self.critic = critic_body
        self.actor_action = nn.Linear(hidden_units[0], config.action_dim)
        self.critic_value = nn.Linear(hidden_units[1], 1)
        self.device = config.device
        self.to(self.device)

    def forward(self, state, action=None):
        phiA = self.actor(state)
        x = self.actor_action(phiA)
        mean = torch.tanh(x)

        phiV = self.critic(state)
        value = self.critic_value(phiV)

        return {'actions': action, 'values': value}
