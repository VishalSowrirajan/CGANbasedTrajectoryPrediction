import os
import numpy as np
import torch

def world2image(traj_w, H_inv):
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    a = np.ones((traj_w.shape[0], 1))
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2])
    return traj_uvz[:, :2].astype(int)

H = np.loadtxt("h.txt")
H_inv = np.linalg.inv(H)
traj = torch.tensor()

converted_traj = []
for tra in traj:
    a = world2image(tra, H_inv)
    converted_traj.append(a)

b = np.asarray(converted_traj)
print(b)