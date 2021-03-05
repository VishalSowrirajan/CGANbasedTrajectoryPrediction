import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3

# 0.4
xSpeed01 = np.asarray([291, 270, 251, 234, 218, 202, 188, 175])
ySpeed01 = np.asarray([329, 319, 311, 306, 304, 303, 302, 302])

xSpeed02 = np.asarray([308, 286, 267, 250, 234, 220, 206, 194])
ySpeed02 = np.asarray([371, 358, 348, 341, 337, 334, 332, 330])

# GT
xSpeedGT01 = np.asarray([257, 230, 206, 182, 155, 131, 105, 79])
ySpeedGT01 = np.asarray([335, 339, 344, 343, 342, 338, 338, 338])

xSpeedGT02 = np.asarray([274, 248, 222, 201, 176, 153, 132, 108])
ySpeedGT02 = np.asarray([380, 380, 373, 373, 367, 362, 362, 362])


# 0.8
xSpeed03 = np.asarray([262, 233, 205, 178, 150, 123, 95, 92])
ySpeed03 = np.asarray([335, 331, 329, 328, 329, 330, 331, 331])

xSpeed04 = np.asarray([280, 251, 225, 198, 172, 145, 119, 92])
ySpeed04 = np.asarray([387, 387, 388, 391, 396, 400, 405, 408])


plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='purple', label='0.4')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='purple')
plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='red', label='0.8')
plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='red')
plt.plot(xSpeedGT01, ySpeedGT01, zorder=1, linewidth=3, linestyle='--', color='blue', label='GT')
plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, linewidth=3, linestyle='--', color='blue')

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='purple')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='purple')


plt.plot(xSpeedGT01[-1], ySpeedGT01[-1], zorder=1, marker='<', color='blue')
plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='<', color='blue')

plt.plot(xSpeed03[-1], ySpeed03[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed04[-1], ySpeed04[-1], zorder=1, marker='<', color='red')


plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel374.png")
plt.imshow(img, zorder=0)
plt.show()
