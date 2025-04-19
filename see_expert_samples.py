import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

#from experts.PG import PG
from cost import CostNN
from utils import to_one_hot, get_cumulative_rewards

from torch.optim.lr_scheduler import StepLR

# SEEDS
seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
plt.figure(figsize=[16, 12])

# ENV SETUP
env_name = 'CartPole-v1'
env = gym.make(env_name,render_mode="human").unwrapped

n_actions = env.action_space.n
state_shape = env.observation_space.shape
if seed is not None:
    state,_ = env.reset(seed=seed)
else:
    state,_ = env.reset()

# LOADING EXPERT/DEMO SAMPLES
demo_trajs = np.load('expert_samples/pg_cartpole.npy', allow_pickle=True)
print(len(demo_trajs))

for i in range(len(demo_trajs)):
    state,_ = env.reset()
    env.state = demo_trajs[i][0][0]  # Set the state of the environment to the first state of the demo trajectory
    for j in range(len(demo_trajs[i][0])):
        env.render()
        _,_,_,_,_ = env.step(demo_trajs[i][1][j])
