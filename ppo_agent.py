import torch
import numpy as np
from operator import itemgetter

from ppo_model import PPOModel
from storage import Storage

class PPOAgent():

    def __init__(self, config):
        self.config = config
        self.env_info = None
        self.env_agents = None
        self.epsilon = 1.0
        self.states = None
        self.batch_size = self.config.config['BatchesSize']
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
        self.train()
    
    def act(self, states):
        # states = torch.from_numpy(states).float().unsqueeze(0).to(self.network.device)
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(0, self.config.action_dim)
        predictions = self.network(states)
        return predictions

    def sample_trajectories(self):
        for t in range(500):
            predictions = self.act(self.states)
            self.env_info = self.config.env.step(predictions['actions'].cpu().numpy())[self.config.brain_name]
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
            self.storage.returns[t] = returns
    
    def train(self):
        indicies_arr = np.arange(len(self.storage.rewards))
        batches = np.random.choice(indicies_arr, size=(10, 50), replace=False)
        actions = torch.cat(self.storage.actions, dim=0)
        states = torch.tensor(self.storage.states).reshape(shape=(-1, 33))
        k = itemgetter(batches[0])(states)
        
        print(k.shape)
        k = states[batches[0]]
        
        print(k.shape)
        
