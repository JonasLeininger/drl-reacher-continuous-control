import yaml

from unityagents import UnityEnvironment
import numpy as np

from ppo_agent import PPOAgent
from ppo_model import PPOModel
from config import Config

def main():
    config = Config()
    config.init_env()
    print_env_information(config)
    run_random_env(config)

    agent = PPOAgent(config)
    train_agent(config, agent)


def load_config_file(config_file: str = 'config/config.yaml'):
    with open(config_file, 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as ex:
            print(ex)


def print_env_information(config):
    env_info = config.env.reset(train_mode=False)[config.brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    print('Size of each action:', config.action_dim)
    config.states = env_info.vector_observations
    print('There are {} agents. Each observes a state with length: {}'.format(config.states.shape[0], config.state_dim))
    print('The state for the first agent looks like:', config.states[0])


def run_random_env(config):
    env_info = config.env.reset(train_mode=False)[config.brain_name]
    num_agents = len(env_info.agents)
    config.states = env_info.vector_observations
    scores = np.zeros(num_agents)
    steps = 1000
    for t in range(steps):
        actions = np.random.randn(num_agents, config.action_dim)
        actions = np.clip(actions, -1, 1)
        env_info = config.env.step(actions)[config.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        config.states = next_states
        if np.any(dones):
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def train_agent(config, agent):
    agent.step()


if __name__=='__main__':
    main()