import torch
import numpy as np
from operator import itemgetter

from base_model import BaseModel
from a2c_model import A2CModel
from storage import Storage

class A2CAgent():

    def __init__(self, config):
        self.config = config
        self.env_info = None
        self.env_agents = None
        self.states = None
        self.dones = None
        self.batch_size = self.config.config['BatchesSize']
        self.storage = Storage(size=2048)
        self.actor_base = BaseModel(config, hidden_units=(128, 64))
        self.critic_base = BaseModel(config, hidden_units=(128, 64))
        self.network = A2CModel(config, self.actor_base, self.critic_base)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.scores = []
        self.scores_agent_mean = []

    def step(self):
        self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
        self.env_agents = self.env_info.agents
        self.states = self.env_info.vector_observations
        self.dones = self.env_info.local_done
        print(self.dones)