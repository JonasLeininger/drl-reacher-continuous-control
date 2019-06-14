import torch
import numpy as np

from ppo_model import PPOModel

class PPOAgent():

    def __init__(self, config):
        self.config = config
        self.env_info = None
        self.env_agents = None
        self.epsilon = 1.0
        self.network = PPOModel(config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

    def step(self):
        self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
        self.env_agents = self.env_info.agents
        states = self.env_info.vector_observations
        predictions = self.act(states)
        self.env_info = self.config.env.step(predictions['action'].cpu().numpy())[self.config.brain_name]
        next_states = self.env_info.vector_observations
        print(next_states.shape)
        rewards = self.env_info.rewards
        dones = self.env_info.local_done

    
    def act(self, states):
        # states = torch.from_numpy(states).float().unsqueeze(0).to(self.network.device)
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(0, self.config.action_dim)
        predictions = self.network(states)
        return predictions

    def sample_trajectories(self):
        for t in range(400):
            states = self.env_info.vector_observations
            predictions = self.act(states)
            self.env_info = self.config.env.step(predictions['action'].cpu().numpy())[self.config.brain_name]
            next_states = self.env_info.vector_observations
            rewards = self.env_info.rewards
