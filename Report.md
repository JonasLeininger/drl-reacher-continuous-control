# Report on solving the Reacher Continuous Control Unity Environment

This is a report on my experiments to solve the reacher environment. The description is in the README. There you can read about downloading the environment, reward per timestep and so on.

## Failed Attemps
I tried to solve the task with the PPO algorithm ([PPO](www.google.com)). The environment converged to a score of +10 and did not evolve from this.

## Solving the Environment with A2C
When I switched to a basic A2C Algorithm the agent finally reached a mean score over the 20 robot arms over 100 episodes of +30.0. With the basic hyperparameter setup from the a2c paper the agent performed bad. So there was quit some tweeking needed.

The agent does not perform stable with my setup all the time. Until now i didn't figured out what hyperparameter to tweek more to get a stable learning agent. Most of the time around episode 70-80 the agent reaches a mean of 30.0 over the 20 robot arms for the first time.