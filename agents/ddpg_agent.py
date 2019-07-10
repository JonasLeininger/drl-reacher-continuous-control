import torch
import numpy as np

from models.base_model import BaseModel
from models.ddpg_actor import DDPGActor
from models.ddpg_critic import DDPGCritic

class DDPGAgent():

    def __init__(self, config):
        self.config = config
        self.checkpoint_path = "checkpoints/ddpg/cp-{epoch:04d}.pt"
        self.episodes = 1000
        self.trajectory_length = 100
        self.env_info = None
        self.env_agents = None
        self.states = None
        self.loss = None
        self.batch_size = self.config.config['BatchesSize']
        self.actor_local = DDPGActor(config)
        self.actor_target = DDPGActor(config)
        self.critic_local = DDPGCritic(config)
        self.critic_target = DDPGCritic(config)
        self.optimizer_actor = torch.optim.Adam(self.actor_local.parameters(), lr=self.config.learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic_local.parameters(), lr=self.config.learning_rate)
        self.scores = []
        self.scores_agent_mean = []

    def run_agent(self):
        for step in range(self.episodes):
            print("Episonde {}/{}".format(step, self.episodes))
            self.env_info = self.config.env.reset(train_mode=True)[self.config.brain_name]
            self.env_agents = self.env_info.agents
            self.states = self.env_info.vector_observations
            self.dones = self.env_info.local_done
            self.run_training()
            print("Average score from 20 agents: >> {:.2f} <<".format(self.scores_agent_mean[-1]))
            if (step+1)%10==0:
                self.save_checkpoint(step+1)
                np.save(file="checkpoints/ddpg/ddpg_save_dump.npy", arr=np.asarray(self.scores))

            if (step + 1) >= 100:
                self.mean_of_mean = np.mean(self.scores_agent_mean[-100:])
                print("Mean of the last 100 episodes: {:.2f}".format(self.mean_of_mean))
                if self.mean_of_mean>=30.0:
                    print("Solved the environment after {} episodes with a mean of {:.2f}".format(step, self.mean_of_mean))
                    np.save(file="checkpoints/ddpg/ddpg_final.npy", arr=np.asarray(self.scores))
                    self.save_checkpoint(step+1)
                    break

    def run_training(self):
        while not np.any(self.dones):
            action_prediction = self.act(torch.tensor(self.states, dtype=torch.float, device=self.actor_local.device))
            print(action_prediction.shape)

    def act(self, state_t):
        return self.actor_local(state_t)
