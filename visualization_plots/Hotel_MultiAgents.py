import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3


xSpeed01 = np.asarray([289, 267, 244, 222, 198])
ySpeed01 = np.asarray([322, 321, 321, 322, 323])

xSpeed02 = np.asarray([207, 238, 264, 291, 317])
ySpeed02 = np.asarray([395, 395, 393, 390, 387])

xSpeed03 = np.asarray([198, 226, 251, 276, 299])
ySpeed03 = np.asarray([424, 422, 419, 416, 411])


plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=2, linestyle='--', color='green', label='Fake_Speed')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=2, linestyle='--', color='green')
plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=2, linestyle='--', color='green')

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='green')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='>', color='green')
plt.plot(xSpeed03[-1], ySpeed03[-1], zorder=1, marker='>', color='green')

plt.legend()
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel382.png")
plt.imshow(img, zorder=0)
plt.show()
