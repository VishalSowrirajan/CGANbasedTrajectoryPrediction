import numpy as np

def to_image_frame(Hinv, loc):
    loc = np.dot(loc, Hinv)  # to camera frame
    return loc / loc[2]  # to pixels (from millimeters)


world_coordinates = [6.8905, 5.7223, 1.]


array_from_file = np.loadtxt("H.txt")
h_transpose = np.transpose(array_from_file)
h_inv =np.linalg.inv(array_from_file)

b = np.dot(h_inv, world_coordinates)
print(np.rint(b))