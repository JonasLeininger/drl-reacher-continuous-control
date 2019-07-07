# Report on solving the Reacher Continuous Control Unity Environment

This is a report on my experiments to solve the reacher environment. The description is in the README. There you can read about downloading the environment, reward per timestep and so on.

## Failed Attemps
I tried to solve the task with the PPO algorithm ([PPO](https://arxiv.org/pdf/1707.06347.pdf)). The environment converged to a score of +10 and did not evolve from this.

## Solving the Environment with A2C
When I switched to a basic A2C Algorithm the agent finally reached a mean score over the 20 robot arms over 100 episodes of +30.0. With the basic hyperparameter setup from the a2c paper the agent performed bad. So there was quit some tweeking needed.

The agent does not perform stable with my setup all the time. Until now i didn't figured out what hyperparameter to tweek more to get a stable learning agent. Most of the time around episode 70-80 the agent reaches a mean of 30.0 over the 20 robot arms for the first time.

In the jupyter notebook the saved weights and scores are loaded. You can perform a test run with the weights and the training performance is displayed in a graph.

If you like to train your own agent with my current setup you can do so in runing the run_a2c.py script.

## Further research
I'm working on an implimentation with the a2c setup as a basis with the ppo algorithm for the training. Until now this algorithm starts learning super fast for the first few episodes and then converges to 0 score or bounces up and down with the score.
I tried a lot of hyperparameters but couldn't figure out what the problem is