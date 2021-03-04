import numpy as np
import matplotlib.pyplot as plt

# GROUP AVOIDANCE PLOT - PLOTTING HOTEL DATASET - 114 second

xSpeed05 = np.asarray([263, 237, 216, 196, 178, 161, 145, 130])
ySpeed05 = np.asarray([384, 371, 360, 352, 347, 342, 338, 334])

xSpeed06 = np.asarray([265, 240, 218, 199, 180, 162, 146, 130])
ySpeed06 = np.asarray([344, 332, 323, 317, 313, 309, 307, 304])

plt.plot(xSpeed05, ySpeed05, zorder=1, color='red', linewidth=2, linestyle='--', label='S=0.6')
plt.plot(xSpeed06, ySpeed06, zorder=1, color='red', linewidth=2, linestyle='--')

plt.plot(xSpeed05[-1], ySpeed05[-1], zorder=1, color='red', marker='<')
plt.plot(xSpeed06[-1], ySpeed06[-1], zorder=1, color='red', marker='<')
img = plt.imread("../hotel115.png")

plt.legend()
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
plt.imshow(img, zorder=0)
plt.show()
