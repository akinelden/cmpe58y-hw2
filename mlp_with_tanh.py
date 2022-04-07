from collections import deque
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
        self.activations.append(x.copy())
        self.outputs.append(x.copy())
        return x

    def backward(self, predictions, targets):
        n_x = predictions.shape[0]
        d_layer_errors = [0 for i in range(len(self.W))]
        d_layer_errors[-1] = 2 * (predictions - targets)
        self.gradW[-1] = (self.outputs[-2].T @ d_layer_errors[-1]) / n_x
        self.gradb[-1] = ( np.ones((1,n_x)) @ d_layer_errors[-1] ) / n_x
        for i in range(-2, -len(self.W), -1): # reverse iterate hidden layers
            d_layer_errors[i] = (1-self.outputs[i]**2)*(d_layer_errors[i+1] @ self.W[i+1].T)
            self.gradW[i] = (self.outputs[i-1].T @ d_layer_errors[i]) / n_x
            self.gradb[i] = (np.ones((1,n_x)) @ d_layer_errors[i]) / n_x
        i-=1
        d_layer_errors[i] = (d_layer_errors[i+1] @ self.W[i+1].T)
        self.gradW[i] = (self.outputs[i-1].T @ d_layer_errors[i]) / n_x 
        self.gradb[i] = (np.ones((1,n_x)) @ d_layer_errors[i]) / n_x 
        return

    def update(self, lr):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - lr * self.gradW[i]
            self.b[i] = self.b[i] - lr * self.gradb[i]
    
    def reset_gradients(self):
        for i in range(len(self.W)):
            self.gradW[i] = np.zeros_like(self.W[i])
            self.gradb[i] = np.zeros_like(self.b[i])

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

    def __init__(self, env, input_dim, hidden_dim, output_dim, num_layers, learning_rate=0.001, gamma=0.99, epsilon=1.0, decay=0.999, replay_memory_size=1000):
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

    def train(self, n_episode, batch_size=4, target_update_steps=64, min_replay_memory=500, min_epsilon=0.1, log = True):
        episode = 0
        while episode < n_episode:
            episode += 1
            score = 0
            done = False
            observation = self.env.reset()
            i = 0
            while not done:
                i += 1
                output = self.training_network.forward(observation.reshape(1,-1))
                action = np.argmax(output) if random.random() > self.epsilon else self.env.action_space.sample()
                new_observation, reward, done, _info = self.env.step(action)
                score += reward
                self.replay_memory.append((observation, action, reward, done, new_observation))
                observation = new_observation
                
                if done:
                    self.scores.append(score)
                    self.epsilon = max(min_epsilon, self.epsilon*self.decay)
                    if log: print(f"Episode {episode} : {score}")

                if len(self.replay_memory) < min(min_replay_memory, self.replay_memory.maxlen): # Then we don't have enough memory, continue playing without learning
                    continue
                
                # learning with mini batches
                batch_indices = np.random.choice(len(self.replay_memory), batch_size, replace=False)
                old_observations, actions, rewards, dones, new_observations = zip(*[self.replay_memory[ind] for ind in batch_indices])
                target_values = [reward_ if done_ else reward_ + self.gamma * np.max(self.target_network.forward(new_obs_.reshape(1,-1))) for new_obs_,reward_,done_ in zip(new_observations, rewards, dones) ]
                predictions = self.training_network.forward(np.array(old_observations))

                targets = predictions.copy()
                for i, action in enumerate(actions) : targets[i, action] = target_values[i]
                self.training_network.backward(predictions, targets)
                self.training_network.update(self.learning_rate)

                if i % target_update_steps == 0:
                    self.target_network.copy_weights_from(self.training_network)
                


        avg_scores = np.mean(np.array(self.scores[len(self.scores) % 10:]).reshape(-1, 10), axis=1)
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].plot(self.scores)
        axs[0].set_title("Individual Scores")
        axs[1].plot(avg_scores)
        axs[1].set_title("10-Averaged Scores")
        plt.show()
    
    def run_visual(self, n_episode):
        score = 0
        episode = 0
        done = False
        observation = self.env.reset()
        while episode < n_episode:
            self.env.render()
            output = self.training_network.forward(observation.reshape(1,-1))
            action = np.argmax(output) if random.random() > self.epsilon else self.env.action_space.sample()
            observation, reward, done, _info = self.env.step(action)
            score += reward
            if done:
                episode+=1
                print(f"Episode {episode} : {score}")
                observation = self.env.reset()
                self.scores.append(score)
                score = 0
                self.epsilon = max(0.1, self.epsilon*self.decay)
