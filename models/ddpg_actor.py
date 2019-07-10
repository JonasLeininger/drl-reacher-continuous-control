import torch
import torch.nn as nn


class DDPGActor(nn.Module):

    def __init__(self, config, hidden_units=(400, 300)):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(config.state_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], config.action_dim)
        self.device = config.device
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x
