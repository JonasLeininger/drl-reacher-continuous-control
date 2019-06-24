import torch
import numpy as np
from operator import itemgetter

from base_model import BaseModel
from a2c_model import A2CModel
from storage import Storage

class A2CAgent():

    def __init__(self, config):
        self.config = config
        self.trajectory_length = 100
        self.env_info = None
        self.env_agents = None
        self.states = None
        self.batch_size = self.config.config['BatchesSize']
        self.storage = Storage(size=11)
        self.actor_base = BaseModel(config, hidden_units=(512, 128))
        self.critic_base = BaseModel(config, hidden_units=(512, 128))
        self.network = A2CModel(config, self.actor_base, self.critic_base)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.scores = []
        self.scores_agent_mean = []

    def run_agent(self):
        self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
        self.env_agents = self.env_info.agents
        self.states = self.env_info.vector_observations
        self.dones = self.env_info.local_done
        self.sample_trajectories()
    
    def act(self, states):
        predictions = self.network(states)
        return predictions
    
    def sample_trajectories(self):
        n_steps = 0
        scores = np.zeros((1, 20))
        l_states = []
        l_dones_num = []
        l_rewards = []
        l_predictions = []
        while not np.any(self.dones):
        # for i in range(20):

            n_steps += 1
            predictions = self.act(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
            l_predictions.append(predictions)
            # print(predictions['actions'])

            self.env_info = self.config.env.step(predictions['actions'].detach()
                            .cpu().numpy())[self.config.brain_name]
            next_states = self.env_info.vector_observations

            self.dones = self.env_info.local_done
            # self.dones[0] = True
            l_dones_num.append(torch.from_numpy(np.vstack([done for done in self.dones if done is not None])
                            .astype(np.uint8)).float().to(self.network.device))
            rewards = self.env_info.rewards
            # print(rewards)
            l_rewards.append(torch.tensor(rewards, dtype=torch.float, device=self.network.device).view(20, 1).detach())
            scores = scores + rewards
            l_states.append(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
            self.states = next_states

            if np.any(self.dones) or (n_steps % self.trajectory_length)==0:
                
                iteration_len = len(l_dones_num)
                l_advantages = [None] * iteration_len
                l_returns = [None] * iteration_len

                advantages = torch.tensor(np.zeros((len(self.env_agents), 1)), dtype=torch.float, device=self.network.device).detach()
                returns = torch.tensor(np.zeros((len(self.env_agents), 1)), dtype=torch.float, device=self.network.device)
                l_advantages[-1] = advantages.detach()
                l_returns[-1] = returns.detach()
                predictions = self.act(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
                l_predictions.append(predictions)

                for k in reversed(range(iteration_len)):
                    returns = l_rewards[k] + 0.99 * (1. - l_dones_num[k]) * returns
                    td_error = l_rewards[k] + 0.99 *(1. - l_dones_num[k]) * l_predictions[k + 1]['values'].detach() - l_predictions[k + 1]['values'].detach()
                    advantages = advantages * 0.99 * 0.95 + td_error
                    l_advantages[k] = advantages.detach()
                    l_returns[k] = returns.detach()
                    
                
                self.train_model(l_predictions, l_advantages, l_returns, l_states)

                l_states = []
                l_dones_num = []
                l_rewards = []
                l_predictions = []
                l_advantages = []
                l_returns = []
        
        print(scores)
        self.scores.append(scores)
        self.scores_agent_mean.append(self.scores[-1].mean())
        print(self.scores_agent_mean[-1])

    def train_model(self, l_predictions, l_advantages, l_returns, l_states):
        # for i in range(5):
        for k in range(len(l_returns)):

            pred = self.network(l_states[k], l_predictions[k]['actions'])
            # pred = l_predictions[k]
            prob_loss = -(pred['log_pi'] * l_advantages[k]) # mean() or sum() any difference?
            value_loss = 0.5 * (l_returns[k] - pred['values']).pow(2)

            self.optimizer.zero_grad()
            (prob_loss + value_loss).mean().backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
