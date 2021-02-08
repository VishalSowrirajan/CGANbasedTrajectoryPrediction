import torch
import numpy as np
import math
import torch.nn as nn

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

m = nn.Sigmoid()

a = torch.tensor([[[-4.8093,  7.6876],
         [ 4.5926,  5.0009],
         [ 4.2694,  4.0365]],
        [[-4.9238,  7.5153],
         [ 4.9054,  4.8598],
         [ 4.4616,  3.8412]],
        [[-5.1312,  7.3013],
         [ 5.1445,  4.6965],
         [ 4.5579,  3.5977]],
        [[-5.4059,  7.0992],
         [ 5.2809,  4.4980],
         [ 4.5508,  3.3006]],
        [[-5.7190,  6.9343],
         [ 5.3131,  4.2539],
         [ 4.4632,  2.9696]],
        [[-6.0490,  6.8039],
         [ 5.2532,  3.9649],
         [ 4.3225,  2.6381]],
        [[-6.3846,  6.7000],
         [ 5.1214,  3.6554],
         [ 4.1476,  2.3282]],
        [[-6.7216,  6.6172],
         [ 4.9407,  3.3560],
         [ 3.9475,  2.0438]],
        [[-7.0589,  6.5518],
         [ 4.7268,  3.0819],
         [ 3.7250,  1.7841]],
        [[-7.3968,  6.5010],
         [ 4.4871,  2.8353],
         [ 3.4809,  1.5504]],
        [[-7.7361,  6.4625],
         [ 4.2252,  2.6167],
         [ 3.2152,  1.3452]],
        [[-8.0774,  6.4347],
         [ 3.9434,  2.4272],
         [ 2.9300,  1.1702]]])

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


