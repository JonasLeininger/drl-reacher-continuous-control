# Report on solving the Reacher Continuous Control Unity Environment

This is a report on my experiments to solve the reacher environment. The description is in the README. There you can read about downloading the environment, reward per timestep and so on.

## Failed Attemps
I tried to solve the task with the PPO algorithm ([PPO](https://arxiv.org/pdf/1707.06347.pdf)). The environment converged to a score of +10 and did not evolve from this.

## Solving the Environment with A2C
When I switched to a basic [A2C](https://arxiv.org/abs/1602.01783) Algorithm the agent finally reached a mean score over the 20 robot arms over 100 episodes of +30.0. With the basic hyperparameter setup from the a2c paper the agent performed bad. So there was quit some tweeking needed.

The agent does not perform stable with my setup all the time. Until now i didn't figured out what hyperparameter to tweek more to always get a stable learning agent. Most of the time around episode 70-80 the agent reaches a mean of 30.0 over the 20 robot arms for the first time.

In the jupyter notebook `report.ipynb` the saved weights and scores are loaded. You can perform a test run with the weights and the training performance is displayed in a graph.

If you like to train your own agent with my current setup you can do so in runing the `run_a2c.py` script.

All my pytorch models are in the `models` folder. For the PPO and A2C I first build the `models/base_model.py` network that consists of two linear layers with 512 and 128 hidden units followed by `relu` activations.
- Linear 512
- relu
- Linear 128
- relu

The A2C is based on actor base model and a critic base model. For the actor the following is added
- Linear layer that maps the 128 hidden units from the second Linear base model to the action dimension
- tanh activation so the actions are in the interval [-1,1]
- a normal distribution layer is added to sample actions and calculate the log probability and the entropy

For the critic one layer with 128 hidden units is added with a mapping to a scalar. The output is parsed without a further activation.

## Solving the Environment with DDPG
[DDPG](https://arxiv.org/abs/1509.02971) solves the task faster and much more stable than the A2C. The switch from A2C to DDPG took some time because of the hyperparameter tuning and the normal frustration that comes with trying different setups. But i learned a lot from switching and the much better results payed off.

Again the report is presented in the `report.ipynb`. I started to refactor the configuration for the a2c and the ddpg setup into a `config.yaml` file.
If you like to train your own agent with my current setup you can do so in runing the `run_ddpg.py` script.

For the DDPG 2 actor and 2 critic networks are initialized. For actor and critic a local and target as we have seen in the double dqn for example.
Actor target and local:
- Linear with mapping from input state dimension for a single agent to 400 hidden units
- batch normalization
- ReLu
- Linear with 300 hidden units
- ReLu
- Linear with mapping from 300 to action dimension for single agent
- tanh activation

Critic target and local:
- Linear with mapping from input state dimension from both agents to 400 hidden units
- batch normalization
- ReLu
- Linear layer with mapping from combination of ReLu output and action dimension of both agents to 300 hidden units
- ReLu
- Linear layer with 300 hidden units
- ReLu


## Further research
I'm working on an implimentation with the a2c setup as a basis with the PPO algorithm for the training. Until now this algorithm starts learning super fast for the first few episodes and then converges to 0 score or bounces up and down with the score.
I tried a lot of hyperparameters but couldn't figure out what the problem is