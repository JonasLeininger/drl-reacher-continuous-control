import torch
import numpy as np

from ppo_model import PPOModel

class PPOAgent():

    def __init__(self, config):
        self.config = config
        self.epsilon = 1.0
        self.network = PPOModel(config)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

    def step(self):
        env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
        states = env_info.vector_observations
        predictions = self.act(states)
        env_info = self.config.env.step(predictions['action'].cpu().numpy())[self.config.brain_name]
        next_states = env_info.vector_observations
        print(next_states.shape)
        rewards = env_info.rewards
        dones = env_info.local_done

    
    def act(self, states):
        # states = torch.from_numpy(states).float().unsqueeze(0).to(self.network.device)
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(0, self.config.action_dim)
        predictions = self.network(states)
        return predictions
