import torch

from ppo_model import PPOModel

class PPOAgent():

    def __init__(self, config):
        self.config = config
        self.network = PPOModel(config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

    def step(self):
        env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
        states = env_info.vector_observations
        predictions = self.network(states)
        print(predictions)
