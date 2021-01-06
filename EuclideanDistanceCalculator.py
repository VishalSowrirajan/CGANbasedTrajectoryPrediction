import torch
import numpy as np
import math
import torch.nn as nn

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

m = nn.Sigmoid()

a = torch.tensor()

distance = []
for a, b in zip(a[:, :], a[1:, :]):
    dist = torch.pairwise_distance(a, b)
    dist = dist.detach().numpy()
    dist = dist.reshape(1, -1)
    distance.append(dist)

distance = np.concatenate(distance, axis=0)
#print(distance)

distance = torch.from_numpy(distance).type(torch.float)
speed = distance / 0.4
print(torch.sigmoid(speed))


