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
        self.activations = [x.copy()]
        self.outputs = [x.copy()]
        for i in range(len(self.W)-1):
            x = x @ self.W[i] + self.b[i]
            self.activations.append(x.copy())
            x = tanh(x)
            self.outputs.append(x.copy())
            # x = np.maximum(x @ self.W[i] + self.b[i], 0)
        x = x @ self.W[-1] + self.b[-1]
        return x

    def backward(self, error, batch_size):
        self.gradW[-1] += (self.outputs[-1].T @ error) / batch_size
        self.gradb[-1] += (1 * error) / batch_size
        hidden_errors = [(1-self.outputs[-1]**2)*(error @ self.W[-1].T)]
        for i in range(len(self.W)-2, 0, -1): # reverse iterate layers
            self.gradW[i] += (self.outputs[i].T @ hidden_errors[0]) / batch_size
            self.gradb[i] += (1 * hidden_errors[0]) / batch_size
            hidden_errors.insert(0, (1-self.outputs[i]**2) * (hidden_errors[0] @ self.W[i].T))
        self.gradW[0] += (self.outputs[0].T @ hidden_errors[0]) / batch_size
        self.gradb[0] += (1 * hidden_errors[0]) / batch_size
        # hidden_errors.insert(0, 1 * (hidden_errors[0] @ self.W[0].T))
        return
        raise NotImplementedError

    def update(self, lr):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - lr * self.gradW[i]
            self.b[i] = self.b[i] - lr * self.gradb[i]
    
    def reset_gradients(self):
        for i in range(len(self.W)):
            self.gradW[i] = np.zeros_like(self.W[i])
            self.gradb[i] = np.zeros_like(self.b[i])

    def copy_weights_from(self, from_net):
        self.W = from_net.W
        self.b = from_net.b

    def __repr__(self):
        return "(" + ", ".join([str(x.shape[0]) for x in self.W]) + ", " + str(self.W[-1].shape[1]) + ")"


class DQN:
    training_network : Network
    target_network : Network
    env : gym.Env
    scores : list[int]

    def __init__(self, env, input_dim, hidden_dim, output_dim, num_layers, learning_rate=0.0001, gamma=0.99, epsilon=1.0, decay=0.999):
        self.env = env
        self.training_network = Network(input_dim, hidden_dim, output_dim, num_layers)
        self.target_network = Network(input_dim, hidden_dim, output_dim, num_layers)
        self.target_network.copy_weights_from(self.training_network)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.scores = []

    def train(self, n_iter, batch_size=16, target_update_steps=48, log = True):
        score = 0
        episode = 0
        done = False
        observation = self.env.reset()
        for i in range(n_iter):
            o = self.training_network.forward(observation.reshape(1,-1))
            action = np.argmax(o) if random.random() > self.epsilon else self.env.action_space.sample()
            observation, reward, done, _info = self.env.step(action)
            score += reward
            if done:
                episode+=1
                if log:
                    print(f"Episode {episode} : {score}")
                target = reward
                observation = self.env.reset()
                self.scores.append(score)
                score = 0
            else:
                target = reward + self.gamma * np.max(self.target_network.forward(observation.reshape(1,-1)))
            error = np.zeros((1,2))
            error[0,action] = 0.5 * (o[0,action] - target)**2
            self.training_network.backward(error, batch_size)
            
            self.epsilon = max(0.1, self.epsilon*self.decay)

            if i % batch_size == 0:
                self.training_network.update(self.learning_rate)
                self.training_network.reset_gradients()
            if i % target_update_steps == 0:
                self.target_network.copy_weights_from(self.training_network)
            

        avg_scores = np.mean(np.array(self.scores[len(self.scores) % 10:]).reshape(-1, 10), axis=1)
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].plot(self.scores)
        axs[0].set_title("Individual Scores")
        axs[1].plot(avg_scores)
        axs[1].set_title("10-Averaged Scores")
        plt.show()
