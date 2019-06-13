import torch

class PPOAgent():

    def __init__(self, model, config):
        self.config = config
        self.network = model
        self.optimizer = torch.optim.Adam()

    def step(self):
