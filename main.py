import gym
import numpy as np
import matplotlib.pyplot as plt
from mlp_with_tanh import *

env = gym.make('CartPole-v0')
env._max_episode_steps = 500
dqn = DQN(env, 4, 16, 2, 1, replay_memory_size=1)
dqn.train(10000,1,64)
dqn.run_visual(20)