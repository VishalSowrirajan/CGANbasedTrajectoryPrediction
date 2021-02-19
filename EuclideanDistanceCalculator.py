import torch
import numpy as np
import math
import torch.nn as nn

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

m = nn.Sigmoid()

a = torch.tensor([[[12.4500,  4.4281],
         [11.1711,  4.1616]],
        [[13.2073,  4.4483],
         [11.9034,  4.2354]],
        [[13.9627,  4.4266],
         [12.6470,  4.2591]],
        [[14.6853,  4.3611],
         [13.3617,  4.2337]],
        [[15.3691,  4.2578],
         [14.0389,  4.1646]],
        [[16.0219,  4.1288],
         [14.6862,  4.0639]],
        [[16.6535,  3.9840],
         [15.3133,  3.9428]],
        [[17.2712,  3.8308],
         [15.9274,  3.8097]],
        [[17.8796,  3.6741],
         [16.5329,  3.6708]],
        [[18.4815,  3.5178],
         [17.1326,  3.5305]],
        [[19.0786,  3.3642],
         [17.7280,  3.3917]],
        [[19.6721,  3.2151],
         [18.3200,  3.2566]]])

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


