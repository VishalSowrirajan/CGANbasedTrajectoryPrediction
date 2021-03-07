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

H = np.loadtxt("hotel_h.txt")
H_inv = np.linalg.inv(H)
traj = torch.tensor([[[ 1.2400, -2.4900],
         [ 2.1600, -2.5400],
         [ 3.1400, -2.3200]],

        [[ 1.1900, -1.8200],
         [ 2.0100, -1.8100],
         [ 3.0900, -1.7000]],

        [[ 1.2700, -1.1700],
         [ 1.9700, -1.1400],
         [ 3.0200, -0.8700]],

        [[ 1.3200, -0.5200],
         [ 1.8900, -0.4800],
         [ 2.8300, -0.2100]],

        [[ 1.3200,  0.1100],
         [ 1.8800,  0.1300],
         [ 2.8100,  0.4200]],

        [[ 1.3300,  0.7200],
         [ 1.8900,  0.7500],
         [ 2.8200,  1.0200]],

        [[ 1.2300,  1.3600],
         [ 1.7900,  1.3900],
         [ 2.9200,  1.6700]],

        [[ 1.1500,  1.9300],
         [ 1.8300,  2.0300],
         [ 3.0900,  2.2500]]])

converted_traj = []
for tra in traj:
    a = world2image(tra, H_inv)
    converted_traj.append(a)

b = np.asarray(converted_traj)
print(b)
for a in range(3):
    lst2 = [item[a] for item in b]
    lt = np.asarray(lst2)
    print(lt[:, 0])
    print(lt[:, 1])