[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# drl-reacher-continuous-control
Deep Reinforcement Learning to solve the Unity ML-Agent [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment

![Trained Agent][image1]

**This environment**
In this environment, a double-jointed arm can move to target locations. A reward of +0.03 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


## Environment

The environment I'm using is precompiled by Udacity. You can download the environment from the links provided below.

## Download Unity Environment

### Version 1: One Agent
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

### Version 2: 20 Agents
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


For training in a cloud like Google Cloud without virtual screen
### Version 1 noVis:
[Linux 1 agent noVis](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)

### Version 2 noVis:
[Linux 20 agents noVis](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)

Place the donwloaded environment into 'UnityEnvs' for example and change the config.yaml in the config folder

## Environment Details
In the run_a2c.py and the a2c_report.ipynb you can see print outs of the environment information. The state vector for each robot arm has a length of 33. To control the arms each needs a action vector of size 4 as input that is in the interval of [-1,1]

Goal of this task is to reach a ball that is on an orbit around the arm and follow the balls movement. Each step the arm's hand is in contact with the environment the arm gets an reward of +0.1. This is at least what the instructions said. But the precompiled environment gave each arm a score of +0.0399 if the hand was in contact with the ball.

The environment is solved if the mean of the arms for each episode shows a mean of 30.0+ over 100 episodes.


## A2C Agent

The a2c_agent.py initiates the a2c model and runs the training of the model for a maximum of 1k episodes with the method 'run_a2c.py'.
The algorithm used is [A2C](https://arxiv.org/pdf/1602.01783.pdf). In the paper the A3C algorithm is used which is asynchronous. The A2C first recieves data from all robot arms and then runs the training loop. In the A3C case the training is done asynchronous.
The Network consists of an actor and a critic network that both are based on the network setup in the base_model.py. The base model is a fully connected network with 2 layers. I found that 512 nodes for the first layer and 128 nodes for the second layer had a good performance. The activation function is relu. I tried tanh in the base_model but this activation function didn't perform well.

In the a2c_model.py the base-actor network needs one more layer to map the states vector to the action vector and the critic needs an additional layer to map to a scalar that is representing V(s). For the action prediction the activation function is tanh so that the actions are between [-1,1]. Also after sampling from the gaussian distribution the actions are forced to be in this interval with the torch.clamp() call.
The forward() path from the a2c_model calculates the predictions for 
- actions
- values
- log_pi
- entropy

The entropy is not used in this algorithm but for other models this can be important (As i saw in my tries to implement the ppo algorithm inside the a2c architecture).

The a2c_agent gathers data from the environment until a preset *trajectory_length* is reached or the episode (1001 steps in the precompiled environment) is finished. Every time the *trajectory_length* or the done state is reached the agent calculates the advantage function for each timestep. I'm using the [GAE](https://arxiv.org/abs/1506.02438) algorithm to calculate this.
To calculate the update step for the actor the log of pi is multiplied with the activations. The loss of the critic is calculated with a standart mean squared error.
For the optimizer I choosed Adam with a learning rate of 1e-4. The run i saved solved the environment after 176 episodes.

In the Report.md you can read about my other failed attemps and in the a2c_report notebook you can see the trained agent in action and see the training performance.