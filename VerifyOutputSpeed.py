import torch
import numpy as np
from constants import *
from utils import get_dataset_name


def get_traj(trajectories, sequences, labels=None):
    print("Enter the sequence you want to visualize from:", sequences)
    seq_start = int(input("Enter the sequence start: "))
    seq_end = int(input("Enter the sequence end:"))
    positions = trajectories[:, seq_start:seq_end, :]
    if MULTI_CONDITIONAL_MODEL:
        label = labels[:, seq_start:seq_end, :]
        return positions, label
    else:
        return positions


def get_distance(trajectories):
    euclid_distance = []
    for a, b in zip(trajectories[:, :], trajectories[1:, :]):
        dist = torch.pairwise_distance(a, b)
        dist = dist.detach().numpy()
        euclid_distance.append(dist.reshape(1, -1))
    euclid_distance = torch.from_numpy(np.concatenate(euclid_distance, axis=0)).type(torch.float)
    return euclid_distance


def inverse_sigmoid(speeds, max_speed=None, labels=None):
    simulated_speed = []
    inv = torch.log((speeds / (1 - speeds)))
    if SINGLE_CONDITIONAL_MODEL:
        print("The current speeds are: ", inv / max_speed)
    else:
        labels = labels.view(PRED_LEN, -1)
        for speed, agent in zip(inv, labels[:PRED_LEN-1, :]):
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
        print("The current speeds are: ", simulated_speed.view(PRED_LEN-1, -1))


def get_speed_from_distance(distance):
    # Since we skip the speed calculation (see trajectories.py for more explanation), we directly pass the distance through sigmoid layer
    sigmoid_speed = torch.sigmoid(distance)
    return sigmoid_speed


def get_max_speed(path):
    if path == "eth":
        return ETH_MAX_SPEED
    elif path == "hotel":
        return HOTEL_MAX_SPEED
    elif path == "zara1":
        return ZARA1_MAX_SPEED
    elif path == "zara2":
        return ZARA2_MAX_SPEED
    elif path == "univ":
        return UNIV_MAX_SPEED


def verify_speed(traj, sequences, labels=None):
    if MULTI_CONDITIONAL_MODEL:
        traj, label = get_traj(traj, sequences, labels=labels)
    else:
        dataset_name = get_dataset_name(SINGLE_TEST_DATASET_PATH)
        traj = get_traj(traj, sequences, labels=None)
    dist = get_distance(traj)
    speed = get_speed_from_distance(dist)
    # We calculate inverse sigmoid to verify the speed
    if MULTI_CONDITIONAL_MODEL:
        inverse_sigmoid(speed, labels=label)
    else:
        maxspeed= get_max_speed(dataset_name)
        inverse_sigmoid(speed, max_speed=maxspeed)
