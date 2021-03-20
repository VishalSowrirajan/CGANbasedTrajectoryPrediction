from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np


def collisionPercentage(traj):
    collided_or_not = []
    no_of_frames = 0
    for _ in range(2):
        curr_Traj = traj[:, 0:3, :].cpu().data.numpy()
        # no_of_frames += curr_frame
        curr_collided_peds = 0
        peds = 0
        for trajectories in curr_Traj:
            peds += trajectories.shape[0]
            dist = squareform(pdist(trajectories, metric="euclidean"))
            np.fill_diagonal(dist, np.nan)
            for rows in dist:
                if any(i <= 2 for i in rows):
                    curr_collided_peds += 1

            percentage_of_collision_in_curr_frame = curr_collided_peds / peds
            peds = 0
            curr_collided_peds = 0
            collided_or_not.append(percentage_of_collision_in_curr_frame)

    collision = sum(collided_or_not) / len(collided_or_not)
    a = sum(collided_or_not)

    return torch.tensor(collision)

example = torch.tensor([[[12.6443,  3.4780],
         [ 3.3576,  5.4893],
         [ 2.8504,  5.0783]],
         [[-2.6978,  6.7272],
         [-2.6954,  7.5725],
         [ 1.4575,  8.6070]]])
seq = torch.tensor([0, 3])
a = collisionPercentage(example)