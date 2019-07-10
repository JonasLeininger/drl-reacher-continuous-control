import torch
import numpy as np

from models.base_model import BaseModel
from models.a2c_model import A2CModel
from storage import Storage


class A2CAgentPPO():

    def __init__(self, config):
        self.config = config
        self.checkpoint_path = "checkpoints/a2c_ppo/cp-{epoch:04d}.pt"
        self.episodes = 1000
        self.trajectory_length = 1000
        self.env_info = None
        self.env_agents = None
        self.states = None
        self.loss = None
        self.batch_size = self.config.config['BatchesSize']
        self.storage = Storage(size=11)
        self.actor_base = BaseModel(config, hidden_units=(512, 128))
        self.critic_base = BaseModel(config, hidden_units=(512, 128))
        self.network = A2CModel(config, self.actor_base, self.critic_base)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.scores = []
        self.scores_agent_mean = []

    def run_agent(self):
        for step in range(self.episodes):
            print("Episonde {}/{}".format(step, self.episodes))
            self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
            self.env_agents = self.env_info.agents
            self.states = self.env_info.vector_observations
            self.dones = self.env_info.local_done
            self.sample_trajectories()
            print("Average score from 20 agents: >> {:.2f} <<".format(self.scores_agent_mean[-1]))
            if (step + 1) % 10 == 0:
                self.save_checkpoint(step + 1)
                np.save(file="checkpoints/a2c_ppo/a2c_ppo_save_dump.npy", arr=np.asarray(self.scores))

            if (step + 1) >= 100:
                self.mean_of_mean = np.mean(self.scores_agent_mean[-100:])
                print("Mean of the last 100 episodes: {:.2f}".format(self.mean_of_mean))
                if self.mean_of_mean >= 30.0:
                    print("Solved the environment after {} episodes with a mean of {:.2f}".format(step,
                                                                                                  self.mean_of_mean))
                    np.save(file="checkpoints/a2c_ppo/a2c_ppo_final.npy", arr=np.asarray(self.scores))
                    self.save_checkpoint(step + 1)
                    break

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

            n_steps += 1
            predictions = self.act(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
            l_predictions.append(predictions)

            self.env_info = self.config.env.step(predictions['actions'].detach()
                                                 .cpu().numpy())[self.config.brain_name]
            next_states = self.env_info.vector_observations

            self.dones = self.env_info.local_done
            l_dones_num.append(torch.from_numpy(np.vstack([done for done in self.dones if done is not None])
                                                .astype(np.uint8)).float().to(self.network.device))
            rewards = self.env_info.rewards
            l_rewards.append(torch.tensor(rewards, dtype=torch.float, device=self.network.device).view(20, 1).detach())
            scores = scores + rewards
            l_states.append(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
            self.states = next_states

            if np.any(self.dones) or (n_steps % self.trajectory_length) == 0:

                iteration_len = len(l_dones_num)
                l_advantages = [None] * iteration_len
                l_returns = [None] * iteration_len

                advantages = torch.tensor(np.zeros((len(self.env_agents), 1)), dtype=torch.float,
                                          device=self.network.device).detach()
                returns = torch.tensor(np.zeros((len(self.env_agents), 1)), dtype=torch.float,
                                       device=self.network.device)
                l_advantages[-1] = advantages.detach()
                l_returns[-1] = returns.detach()
                predictions = self.act(torch.tensor(self.states, dtype=torch.float, device=self.network.device))
                l_predictions.append(predictions)

                for k in reversed(range(iteration_len)):
                    returns = l_rewards[k] + 0.99 * (1. - l_dones_num[k]) * returns
                    td_error = l_rewards[k] + 0.99 * (1. - l_dones_num[k]) * l_predictions[k + 1]['values'].detach() - \
                               l_predictions[k + 1]['values'].detach()
                    advantages = advantages * 0.99 * 0.95 + td_error
                    l_advantages[k] = advantages.detach()
                    l_returns[k] = returns.detach()

                # print(len(l_advantages))
                # print(l_advantages[0].shape)
                vec_advantages = torch.cat(l_advantages).detach()
                vec_advantages = vec_advantages.view(iteration_len, 20, 1)
                _advantages = (vec_advantages - vec_advantages.mean()) / vec_advantages.std()
                # print(_advantages.shape)
                # print(_advantages[0].shape)
                self.train_model(l_predictions, _advantages, l_returns, l_states)

                l_states = []
                l_dones_num = []
                l_rewards = []
                l_predictions = []
                l_advantages = []
                l_returns = []

        print("Agent scores:")
        print(scores)
        self.scores.append(scores)
        self.scores_agent_mean.append(scores.mean())

    def train_model(self, l_predictions, l_advantages, l_returns, l_states):
        for k in range(len(l_returns)):
            pred = self.network(l_states[k], l_predictions[k]['actions'])
            ratio = (pred['log_pi'] - l_predictions[k]['log_pi']).exp()
            ratio_clip = ratio.clamp(1.0 - 0.2, 1.0 + 0.2)
            policy_loss = - torch.min(ratio * l_advantages[k],
                                      ratio_clip * l_advantages[k]).mean() - pred['entropy'].mean() * 0.01

            value_loss = 0.5 * (l_returns[k] - pred['values']).pow(2).mean()

            self.loss = (policy_loss + value_loss)
            self.optimizer.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

    def save_checkpoint(self, epoch: int):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, self.checkpoint_path.format(epoch=epoch))

    def load_checkpoint(self, checkpoint: str):
        checkpoint = torch.load(checkpoint)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']

        self.network.eval()