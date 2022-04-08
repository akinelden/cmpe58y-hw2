from collections import deque
import pandas as pd
import numpy as np
import random
import gym
import matplotlib.pyplot as plt

def tanh(x):
    if (type(x) == int) or (type(x) == float):
        if x > 30:
            return 1.0
        elif x < -30:
            return -1.0
        else:
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    elif type(x) == np.ndarray:
        act = np.zeros_like(x)
        act = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        act[x>30] = 1.0
        act[x<-30] = -1.0
    return act

def dtanh(x):
    return 1-np.power(tanh(x), 2)

class Network:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        self.W = [np.random.randn(input_dim, hidden_dim)*((2/(input_dim+hidden_dim))**0.5)]
        self.b = [np.zeros(hidden_dim)]
        self.gradW = [np.zeros((input_dim, hidden_dim))]
        self.gradb = [np.zeros((1,hidden_dim))]
        for i in range(num_layers):
            self.W.append(np.random.randn(hidden_dim, hidden_dim)*((1/hidden_dim)**0.5))
            self.b.append(np.zeros(hidden_dim))
            self.gradW.append(np.zeros((hidden_dim, hidden_dim)))
            self.gradb.append(np.zeros((1,hidden_dim)))
        self.W.append(np.random.randn(hidden_dim, output_dim)*((2/(hidden_dim+output_dim))**0.5))
        self.b.append(np.zeros(output_dim))
        self.gradW.append(np.zeros((hidden_dim, output_dim)))
        self.gradb.append(np.zeros((1,output_dim)))
        self.activations = []
        self.outputs = []

    def forward(self, x):
        # keep both activations and outputs in the memory
        self.activations = [x.copy()]
        self.outputs = [x.copy()]
        for i in range(len(self.W)-1):
            x = x @ self.W[i] + self.b[i]
            self.activations.append(x.copy())
            x = tanh(x)
            self.outputs.append(x.copy())
            # x = np.maximum(x @ self.W[i] + self.b[i], 0)
        x = x @ self.W[-1] + self.b[-1]
        self.activations.append(x.copy())
        self.outputs.append(x.copy())
        return x

    def backward(self, predictions, targets):
        # n_x is the batch size
        n_x = predictions.shape[0]
        # keep layer errors
        layer_errors = [0 for i in range(len(self.W))]
        # output layer error is 2 * (y_pred - y_targ)
        layer_errors[-1] = 2 * (predictions - targets)
        self.gradW[-1] = (self.outputs[-2].T @ layer_errors[-1]) / n_x
        self.gradb[-1] = ( np.ones((1,n_x)) @ layer_errors[-1] ) / n_x
        for i in range(-2, -len(self.W)-1, -1): # reverse iterate hidden layers
            # since derivative of tanh(x) is (1-tanh(x)**2) directly use output instead of tanh(activation)  
            layer_errors[i] = (1-self.outputs[i]**2)*(layer_errors[i+1] @ self.W[i+1].T)
            self.gradW[i] = (self.outputs[i-1].T @ layer_errors[i]) / n_x
            self.gradb[i] = (np.ones((1,n_x)) @ layer_errors[i]) / n_x
        return

    def update(self, lr):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - lr * self.gradW[i]
            self.b[i] = self.b[i] - lr * self.gradb[i]

    def copy_weights_from(self, from_net):
        self.W = from_net.W.copy()
        self.b = from_net.b.copy()

    def __repr__(self):
        return "(" + ", ".join([str(x.shape[0]) for x in self.W]) + ", " + str(self.W[-1].shape[1]) + ")"


class DQN:
    training_network : Network
    target_network : Network
    replay_memory: deque[tuple]
    env : gym.Env
    scores : list[int]

    def __init__(self, env, input_dim, hidden_dim, output_dim, num_layers, learning_rate=0.0001, gamma=0.99, epsilon=1.0, decay=0.999, replay_memory_size=1000):
        self.env = env
        self.training_network = Network(input_dim, hidden_dim, output_dim, num_layers)
        self.target_network = Network(input_dim, hidden_dim, output_dim, num_layers)
        self.target_network.copy_weights_from(self.training_network)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.scores = []
        self.replay_memory = deque(maxlen=replay_memory_size)

    def train(self, n_episode, batch_size=4, target_update_steps=64, min_epsilon=0.1, log = True):
        batch_size = min(batch_size, self.replay_memory.maxlen)
        episode = 0
        iter = 0
        while episode < n_episode:
            # initialize for each episode:
            episode += 1
            score = 0
            done = False
            observation = self.env.reset()
            while not done:
                # for each step in episode
                iter += 1
                # first forward pass using current observation
                output = self.training_network.forward(observation)
                # then select action with epsilon greedy
                action = np.argmax(output) if random.random() > self.epsilon else self.env.action_space.sample()
                # perform action and observe new state
                new_observation, reward, done, _info = self.env.step(action)
                score += reward
                # add state information to replay memory
                self.replay_memory.append((observation, action, reward, done, new_observation))
                # set current observation as the new observation
                observation = new_observation
                
                if done:
                    # if episode is ended, add score and update epsilon
                    self.scores.append(score)
                    self.epsilon = max(min_epsilon, self.epsilon*self.decay)
                    if log: print(f"Episode {episode} : {score}")
                # If we don't have enough memory, continue playing without learning
                if len(self.replay_memory) < self.replay_memory.maxlen: 
                    continue
                
                # select random sample from replay memory
                old_observations, actions, rewards, dones, new_observations = zip(*random.sample(self.replay_memory, batch_size))
                
                # get the q value of new state by using target network
                q_news = np.max(self.target_network.forward(np.array(new_observations)), axis=1)
                # get the target value as reward or reward + q_new according to done condition
                target_values = np.array(rewards) + self.gamma * (1-np.array(dones)) * q_news

                # get the predictions using training network
                predictions = self.training_network.forward(np.array(old_observations))
                
                # copy predictions as targets, and update the values corresponding to actions with the 
                # above calculated target values
                targets = predictions.copy()
                targets[np.arange(len(actions)), np.array(actions)] = target_values
                
                # apply backpropagation and update the network
                self.training_network.backward(predictions, targets)
                self.training_network.update(self.learning_rate)

                # update target network if enough steps are taken
                if iter % target_update_steps == 0:
                    self.target_network.copy_weights_from(self.training_network)

    def plot_scores(self, roll_size=100):
        data = pd.DataFrame({"scores" : self.scores})
        data["avg"] = data.scores.rolling(roll_size).mean()
        plt.figure(figsize=(12,6))
        plt.xlabel = "Episode"
        plt.ylabel = "Score"
        plt.plot(data["scores"], color="red", alpha=0.2)
        plt.plot(data["avg"], color="blue")
        plt.show()
    
    def run_visual(self, n_episode):
        score = 0
        episode = 0
        done = False
        observation = self.env.reset()
        while episode < n_episode:
            self.env.render()
            output = self.training_network.forward(observation)
            action = np.argmax(output) if random.random() > self.epsilon else self.env.action_space.sample()
            observation, reward, done, _info = self.env.step(action)
            score += reward
            if done:
                episode+=1
                print(f"Episode {episode} : {score}")
                observation = self.env.reset()
                self.scores.append(score)
                score = 0
