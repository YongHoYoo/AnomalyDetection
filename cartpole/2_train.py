# Solution of Open AI gym environment "Cartpole-v0" (https://gym.openai.com/envs/CartPole-v0) using DQN and Pytorch.
# It is is slightly modified version of Pytorch DQN tutorial from
# http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
# The main difference is that it does not take rendered screen as input but it simply uses observation values from the \
# environment.

import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# hyper parameters
EPISODES = 1000  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.01  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.80  # Q-learning discount factor
LR = 0.0005  # NN optimizer learning rate
HIDDEN_LAYER = 24  # NN hidden layer size
BATCH_SIZE = 128  # Q-learning batch size

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition): # what is transition??
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 16)
        self.l3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

def select_action(state):
    sample = random.random() # 0 ~ 1 
    return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)


def run_episode(e, environment):
    state = environment.reset()
    steps = 0

    all_states = [] 

    while True:
        steps += 1
        environment.render()
        action = select_action(FloatTensor([state]))

        all_states.append(state)
        next_state, _, done, _ = environment.step(action[0, 0].item())

        state = next_state

        if done:
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            episode_durations.append(steps)
            plot_durations()
            break
    return torch.tensor(all_states) 

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.1)  # pause a bit so that plots are updated


env = gym.make('CartPole-v0') 
env = wrappers.Monitor(env, './tmp/cartpole-v0-1', force=True) 

model = Network() 

checkpoint = torch.load('cartpole_model.pt') 
model.load_state_dict(checkpoint['state_dict']) 

if use_cuda:
    model.cuda()

TRAIN_SIZE = 70 
VALID_SIZE = 30
TEST_SIZE = 30

episode_durations = []

trainset = [] 
for t in range(TRAIN_SIZE): 
    trainset.append(run_episode(t, env)) 

validset = [] 
for t in range(VALID_SIZE):
    trainset.append(run_episode(t, env)) 


