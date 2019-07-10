import time

import numpy as np
import torch

from agents.a2c_agent import A2CAgent
from config.config import Config

def main():
    config = Config()
    print_env_information(config)
    run_random_env(config)

    agent = A2CAgent(config)
    agent.load_checkpoint('weights/a2c_simple/cp-0050.pt')
    run_agent_model(config, agent)
    agent.load_checkpoint('weights/a2c_simple/cp-0100.pt')
    run_agent_model(config, agent)
    agent.load_checkpoint('weights/a2c_simple/cp-0176.pt')
    run_agent_model(config, agent)

def run_agent_model(config, agent):
    env_info = config.env.reset(train_mode=False)[config.brain_name]
    states = env_info.vector_observations
    scores = np.zeros(config.num_agents)
    steps = 500
    for t in range(steps):
        prediction = agent.network(torch.tensor(states, dtype=torch.float, device=agent.network.device))
        env_info = config.env.step(prediction['actions'].detach().cpu().numpy())[config.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        time.sleep(0.01)
        if np.any(dones):
            break


def print_env_information(config):
    config.env_info = config.env.reset(train_mode=False)[config.brain_name]
    config.num_agents = len(config.env_info.agents)
    print('Number of agents:', config.num_agents)
    print('Size of each action:', config.action_dim)
    config.states = config.env_info.vector_observations
    print('There are {} agents. Each observes a state with length: {}'.format(config.states.shape[0], config.state_dim))
    print('The state for the first agent looks like:', config.states[0])


def run_random_env(config):
    env_info = config.env.reset(train_mode=False)[config.brain_name]
    states = env_info.vector_observations
    scores = np.zeros(config.num_agents)
    steps = 100
    for t in range(steps):
        actions = np.random.randn(config.num_agents, config.action_dim)
        actions = np.clip(actions, -1, 1)
        env_info = config.env.step(actions)[config.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        time.sleep(0.01)
        if np.any(dones):
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))



if __name__=='__main__':
    main()