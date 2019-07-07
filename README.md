# drl-reacher-continuous-control
Deep Reinforcement Learning to solve the Unity ML-Agent Reacher environment

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

## A2C Agent

The a2c_agent.py initiates the a2c model and runs the training of the model for a maximum of 1k episodes with the method 'run_agent()'.