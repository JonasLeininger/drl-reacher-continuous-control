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
        self.scores = []
        self.scores_agent_mean = []

    def step(self):
        self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
        self.env_agents = self.env_info.agents
        self.states = self.env_info.vector_observations
        self.sample_trajectories()
        self.calculate_returns()
        print(self.storage.returns[0])
        print(self.storage.returns[499])
        self.train()
        print(self.scores[-1])
        print(self.scores_agent_mean[-1])
        self.storage.reset()
    
    def act(self, states):
        # states = torch.from_numpy(states).float().unsqueeze(0).to(self.network.device)
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(0, self.config.action_dim)
        predictions = self.network(states)
        return predictions

    def sample_trajectories(self):
        for t in range(500):
            predictions = self.act(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
            self.env_info = self.config.env.step(predictions['actions'].cpu().numpy())[self.config.brain_name]
            next_states = self.env_info.vector_observations
            rewards = self.env_info.rewards
            self.storage.add(predictions)
            self.storage.add({'rewards': rewards,
                              'states': self.states
                              })
            self.states = next_states

        predictions = self.network(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
        self.storage.add(predictions)
        self.storage.placeholder()
        
    def calculate_returns(self):
        self.storage.returns[-1] = np.asarray(self.storage.rewards[-1])
        for t in reversed(range(499)):
            self.storage.returns[t] = self.storage.returns[t+1] + np.asarray(self.storage.rewards[t])
        self.scores.append(self.storage.returns[0])
        self.scores_agent_mean.append(np.mean(self.storage.returns[0]))
    
    def train(self):
        indicies_arr = np.arange(len(self.storage.rewards))
        batches = np.random.choice(indicies_arr, size=(10, 50), replace=False)
        actions = torch.cat(self.storage.actions, dim=0).detach()
        log_pi_old = torch.cat(self.storage.log_pi, dim=0).detach()
        states = torch.tensor(self.storage.states, dtype=torch.float, device=self.network.device).reshape(shape=(-1, 33)).detach()
        returns = torch.tensor(self.storage.returns, dtype=torch.float, device=self.network.device).reshape(shape=(-1, 1))
        for batch in range(batches.shape[0]):
            indicies = batches[batch]
            sample_actions = actions[indicies]
            sample_log_pi_old = log_pi_old[indicies]
            sample_states = states[indicies]
            sample_returns = returns[indicies]

            prediction = self.network(sample_states, sample_actions)
            ratio = (prediction['log_pi'] - sample_log_pi_old).exp()
            ratio_clip = ratio.clamp(1.0 - 0.2, 1.0 + 0.2)
            policy_loss = - torch.min(ratio * sample_returns, ratio_clip * sample_returns).mean() - \
                          prediction['entropy'].mean() * 0.01

            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
