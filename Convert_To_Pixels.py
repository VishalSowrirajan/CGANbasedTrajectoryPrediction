import numpy as np

def to_image_frame(Hinv, loc):
    loc = np.dot(loc, Hinv)  # to camera frame
    return loc / loc[2]  # to pixels (from millimeters)

world_coordinates = [13.4487205051,	3.93788669527, 1.]

array_from_file = np.loadtxt("h.txt")
h_inv =np.linalg.inv(array_from_file)

traj_cam = np.matmul(h_inv, world_coordinates)
traj_uvz = traj_cam/traj_cam[2]
print(traj_uvz)