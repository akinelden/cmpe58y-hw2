import gym
import numpy as np
import matplotlib.pyplot as plt
from mlp_with_tanh import *

env = gym.make('CartPole-v0')
dqn = DQN(env, 4, 16, 2, 1)
dqn.train(100000)