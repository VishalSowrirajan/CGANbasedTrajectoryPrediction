import torch
import numpy as np
import math
import torch.nn as nn

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

m = nn.Sigmoid()

a = torch.tensor([[[8.1697, 6.8132],
         [8.3095, 7.3111],
         [8.2969, 4.3797]],
        [[8.2818, 6.7721],
         [8.4370, 7.2267],
         [8.3914, 4.3640]],
        [[8.3344, 6.7751],
         [8.5007, 7.1969],
         [8.4323, 4.3863]],
        [[8.3555, 6.7907],
         [8.5289, 7.1892],
         [8.4469, 4.4152]],
        [[8.3658, 6.8050],
         [8.5434, 7.1871],
         [8.4540, 4.4385]],
        [[8.3741, 6.8141],
         [8.5540, 7.1846],
         [8.4610, 4.4540]],
        [[8.3826, 6.8185],
         [8.5640, 7.1806],
         [8.4689, 4.4632]],
        [[8.3910, 6.8197],
         [8.5736, 7.1755],
         [8.4766, 4.4681]],
        [[8.3983, 6.8189],
         [8.5825, 7.1701],
         [8.4832, 4.4702]],
        [[8.4041, 6.8171],
         [8.5900, 7.1647],
         [8.4881, 4.4707]],
        [[8.4084, 6.8148],
         [8.5963, 7.1595],
         [8.4914, 4.4702]],
        [[8.4114, 6.8123],
         [8.6015, 7.1547],
         [8.4934, 4.4692]]])

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
#speed = distance
print(torch.sigmoid(speed))


