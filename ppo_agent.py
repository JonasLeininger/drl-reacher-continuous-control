import torch
import numpy as np

from ppo_model import PPOModel
from storage import Storage

class PPOAgent():

    def __init__(self, config):
        self.config = config
        self.env_info = None
        self.env_agents = None
        self.epsilon = 1.0
        self.states = None
        self.storage = Storage(size=500)
        self.network = PPOModel(config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

    def step(self):
        self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
        print(self.env_info)
        self.env_agents = self.env_info.agents
        self.states = self.env_info.vector_observations
        self.sample_trajectories()
        self.calculate_returns()
    
    def act(self, states):
        # states = torch.from_numpy(states).float().unsqueeze(0).to(self.network.device)
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(0, self.config.action_dim)
        predictions = self.network(states)
        return predictions

    def sample_trajectories(self):
        for t in range(500):
            predictions = self.act(self.states)
            self.env_info = self.config.env.step(predictions['action'].cpu().numpy())[self.config.brain_name]
            next_states = self.env_info.vector_observations
            rewards = self.env_info.rewards
            self.storage.add(predictions)
            self.storage.add({'rewards': rewards,
                              'states': self.states
                              })
            self.states = next_states

        predictions = self.network(self.states)
        self.storage.add(predictions)
        self.storage.placeholder()
        
    
    def calculate_returns(self):
        for t in reversed(range(500)):
            returns = self.storage.rewards[t]
            self.storage.returns[t]
