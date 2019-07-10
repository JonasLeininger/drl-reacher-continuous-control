import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGCritic(nn.Module):

    def __init__(self, config, hidden_units=(400, 300)):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(config.state_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        self.device = config.device
        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = torch.relu(x)
        mu = torch.cat((x, action), dim=1)
        mu = torch.relu(mu)
        mu = self.fc2(mu)
        mu = torch.relu(mu)
        mu = self.fc3(mu)
        mu = torch.relu(mu)
        return mu
