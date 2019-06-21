import torch
import numpy as np
from operator import itemgetter

from base_model import BaseModel
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
        self.storage = Storage(size=2048)
        self.actor_base = BaseModel(config)
        self.critic_base = BaseModel(config)
        self.network = PPOModel(config, self.actor_base, self.critic_base)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.scores = []
        self.scores_agent_mean = []

    def step(self):
        self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
        self.env_agents = self.env_info.agents
        self.states = self.env_info.vector_observations
        self.sample_trajectories()
        self.calculate_returns()
        self.train()
        self.scores_agent_mean.append(self.scores[-1].mean())
        print(self.scores_agent_mean[-1])
        self.storage.reset()
    
    def act(self, states):
        # states = torch.from_numpy(states).float().unsqueeze(0).to(self.network.device)
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(0, self.config.action_dim)
        predictions = self.network(states)
        return predictions

    def sample_trajectories(self):
        for t in range(2048):
            predictions = self.act(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
            self.env_info = self.config.env.step(predictions['actions'].cpu().numpy())[self.config.brain_name]
            next_states = self.env_info.vector_observations
            rewards = self.env_info.rewards
            self.storage.add(predictions)
            self.storage.add({'rewards': torch.tensor(rewards, dtype=torch.float, device=self.network.device).unsqueeze(-1),
                              'states': torch.tensor(self.states, dtype=torch.float, device=self.network.device)
                              })
            self.states = next_states
        
        predictions = self.network(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
        self.storage.add(predictions)
        self.storage.placeholder()
        
    def calculate_returns(self):
        advantages = torch.tensor(np.zeros((len(self.env_agents), 1)), dtype=torch.float, device=self.network.device)
        returns = self.storage.values[-1].detach()
        self.storage.advantages[-1] = advantages.detach()
        self.storage.returns[-1] = returns
        scores = torch.tensor(np.zeros((20, 1)), dtype=torch.float, device=self.network.device)
        for t in reversed(range(2048)):
            returns = self.storage.rewards[t] + 0.99 * returns
            td_error = self.storage.rewards[t] + 0.99* self.storage.values[t+1] - self.storage.values[t]
            advantages = advantages*0.99 + td_error
            self.storage.advantages[t] = advantages.detach()
            self.storage.returns[t] = returns.detach()
            scores = scores + self.storage.rewards[t].detach()
        self.scores.append(scores.detach().cpu().numpy())
        
    
    def train(self):
        indicies_arr = np.arange(len(self.storage.rewards))
        batches = np.random.choice(indicies_arr, size=(32, 64), replace=False)
        actions = torch.cat(self.storage.actions, dim=0).detach()
        log_pi_old = torch.cat(self.storage.log_pi, dim=0).detach()
        states = torch.cat(self.storage.states ).detach()
        returns = torch.cat(self.storage.returns).detach()
        advantages = torch.cat(self.storage.advantages).detach()
        for batch in range(batches.shape[0]):
            indicies = batches[batch]
            sample_actions = actions[indicies]
            sample_log_pi_old = log_pi_old[indicies]
            sample_states = states[indicies]
            sample_returns = returns[indicies]
            sample_advantages = advantages[indicies]

            prediction = self.network(sample_states, sample_actions)
            ratio = (prediction['log_pi'] - sample_log_pi_old).exp()
            ratio_clip = ratio.clamp(1.0 - 0.2, 1.0 + 0.2)
            policy_loss = - torch.min(ratio * sample_advantages, ratio_clip * sample_advantages).mean() - \
                          prediction['entropy'].mean() * 0.01
            
            value_loss = 0.5 * (sample_returns - prediction['values']).pow(2).mean()

            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
