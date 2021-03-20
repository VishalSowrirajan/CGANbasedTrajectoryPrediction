import numpy as np
import matplotlib.pyplot as plt

xSpeed01 = np.asarray([457, 423, 390, 360, 330, 301, 272, 243])
ySpeed01 = np.asarray([347, 348, 350, 351, 352, 353, 354, 355])

xSpeed02 = np.asarray([472, 437, 404, 374, 344, 315, 286, 257])
ySpeed02 = np.asarray([313, 314, 315, 316, 317, 318, 319, 320])

xSpeedGT01 = np.asarray([464, 428, 391, 353, 315, 276, 237, 204])
ySpeedGT01 = np.asarray([343, 343, 346, 346, 348, 349, 352, 358])

xSpeedGT02 = np.asarray([476, 440, 400, 361, 329, 291, 252, 214])
ySpeedGT02 = np.asarray([312, 312, 314, 317, 317, 320, 321, 325])


plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red', label='0.9')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='red')

plt.plot(xSpeedGT01, ySpeedGT01, zorder=1, linewidth=3, color='blue', label='GT', linestyle='--')
plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, linewidth=3, color='blue', linestyle='--')#

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='red')

plt.plot(xSpeedGT01[-1], ySpeedGT01[-1], zorder=1, marker='<', color='blue')
plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='<', color='blue')


#plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel8.png")
plt.imshow(img, zorder=0)
plt.show()
