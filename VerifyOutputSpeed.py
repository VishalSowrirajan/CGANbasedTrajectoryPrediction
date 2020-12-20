import torch
import numpy as np
from constants import *


def get_traj(trajectories, sequences, labels):
    print("Enter the sequence you want to visualize from:", sequences)
    seq_start = int(input("Enter the sequence start: "))
    seq_end = int(input("Enter the sequence end:"))
    positions = trajectories[:, seq_start:seq_end, :]
    label = labels[seq_start:seq_end, :]
    return positions, label


def get_distance(trajectories):
    euclid_distance = []
    for a, b in zip(trajectories[:, :], trajectories[1:, :]):
        dist = torch.pairwise_distance(a, b)
        dist = dist.detach().numpy()
        euclid_distance.append(dist.reshape(1, -1))
    euclid_distance = torch.from_numpy(np.concatenate(euclid_distance, axis=0)).type(torch.float)
    return euclid_distance


def inverse_sigmoid(speeds, labels):
    simulated_speed = []
    inv = torch.log((speeds / (1 - speeds)))
    print(inv)
    for speed in inv:
        for a, b, in zip(speed, labels):
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
    print("The current speeds are: ", simulated_speed.view(PRED_LEN-1, -1))


def get_speed_from_distance(distance):
    # Since we skip the speed calculation (see trajectoreis.py for more explanation), we directly pass the distance through sigmoid layer
    sigmoid_speed = torch.sigmoid(distance)
    return sigmoid_speed


def verify_speed(traj, sequences, labels):
    traj, label = get_traj(traj, sequences, labels)
    dist = get_distance(traj)
    speed = get_speed_from_distance(dist)
    # We calculate inverse sigmoid to verify the speed
    inverse_sigmoid(speed, label)
