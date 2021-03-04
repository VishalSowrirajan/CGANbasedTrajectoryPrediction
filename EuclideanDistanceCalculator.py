import torch
import numpy as np
import math
import torch.nn as nn
from constants import *

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

m = nn.Sigmoid()

a = torch.tensor()

def get_speeds_multiagent(traj, labels):
    distance = []
    for a, b in zip(traj[:, :], traj[1:, :]):
        dist = torch.pairwise_distance(a, b)
        dist = dist.detach().numpy()
        dist = dist.reshape(1, -1)
        distance.append(dist)
    distance = torch.from_numpy(np.concatenate(distance, axis=0)).type(torch.float)
    sigmoid_speed = torch.sigmoid(distance)
    inv = torch.log((sigmoid_speed / (1 - sigmoid_speed)))
    simulated_speed = []
    labels = labels.view(PRED_LEN, -1)
    for speed, agent in zip(inv, labels[:PRED_LEN - 1, :]):
        for a, b, in zip(speed, agent):
            if torch.eq(b, 0.1):
                s = a / AV_MAX_SPEED
                simulated_speed.append(s.view(1, 1))
            elif torch.eq(b, 0.2):
                s = a / OTHER_MAX_SPEED
                simulated_speed.append(s.view(1, 1))
            elif torch.eq(b, 0.3):
                s = a / AGENT_MAX_SPEED
                simulated_speed.append(s.view(1, 1))
    simulated_speed = torch.cat(simulated_speed, dim=0)
    print('the labels are: ', labels)
    print("The current speeds are: ", simulated_speed.view(PRED_LEN - 1, -1))



