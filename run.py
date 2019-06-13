import yaml

from unityagents import UnityEnvironment
import numpy as np

def main():
    config = load_config_file()
    env = UnityEnvironment(file_name=config['Reacher20'])

    print_env_information(env)
    run_random_env(env)


def load_config_file(config_file: str = 'config/config.yaml'):
    with open(config_file, 'r') as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as ex:
            print(ex)


def print_env_information(env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


def run_random_env(env):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]
    scores = np.zeros(num_agents)
    steps = 1000
    for t in range(steps):
        actions = np.random.randn(num_agents, action_size)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


def train_agent(env, agent):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]


if __name__=='__main__':
    main()