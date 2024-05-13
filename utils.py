import math
import random
from collections import namedtuple, deque
from torch.utils.data import Dataset
import torch
import numpy as np

import stable_baselines3 as sb3
import crafter

env = crafter.Env()
model = sb3.PPO('CnnPolicy', env, verbose=1)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'skill', 'reward', 'init_terminal'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
def epsilon(start, end, n, num_episodes, min=0):
    eps = start - (start-end)*n/num_episodes
    if eps > min:
        return eps
    else:
        return min

class MemoryDataset(Dataset):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = 0
        self.memory = deque([], maxlen=capacity)

    def add(self, data):
        if len(self.memory):
            a = self.memory[0]
        self.memory.append(data)
        b = self.memory[0]
        if len(self.memory) == self.capacity:
            assert a != b

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx][0], self.memory[idx][1]
    
class EarlyStopper:
    def __init__(self, if_save=False, patience=1, min_delta=0):
        # if_save: whether save the best model so far
        # patience: stop training after ```patience``` times iterations
        # min_delta: tolerance when comparing losses
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.if_save = if_save

    def early_stop(self, validation_loss, state_dict=None, path=None):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if state_dict and self.if_save and path:
                torch.save(state_dict, path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# def plot_durations(show_result=False):
#     plt.figure(1)
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     # if len(durations_t) >= 100:
#     #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#     #     means = torch.cat((torch.zeros(99), means))
#     #     plt.plot(means.numpy())

#     plt.pause(0.01)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())