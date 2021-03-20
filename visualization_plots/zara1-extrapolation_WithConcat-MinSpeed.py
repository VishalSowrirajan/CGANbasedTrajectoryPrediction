import numpy as np
import matplotlib.pyplot as plt

xSpeed01 = np.asarray([244, 260, 272, 281, 287, 292, 296, 298, 300, 302, 303, 304])
ySpeed01 = np.asarray([373, 371, 369, 367, 364, 362, 360, 358, 356, 354, 352, 349])

xSpeed02 = np.asarray([241, 257, 269, 278, 285, 289, 293, 296, 298, 299, 301, 302])
ySpeed02 = np.asarray([343, 342, 340, 338, 336, 334, 331, 329, 327, 325, 322, 320])

xSpeed03 = np.asarray([452, 438, 428, 420, 414, 409, 405, 401, 398, 395, 393, 390])
ySpeed03 = np.asarray([382, 384, 386, 387, 388, 388, 388, 387, 387, 386, 385, 385])

xSpeed04 = np.asarray([457, 443, 432, 424, 418, 413, 409, 405, 402, 399, 396, 394])
ySpeed04 = np.asarray([414, 416, 416, 416, 416, 415, 414, 413, 411, 410, 408, 407])


plt.plot(xSpeed01,ySpeed01,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='yellow', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='purple', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='orange', linewidth=3, linestyle='--')

plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='>', color='red', linewidth=2)
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='>', color='yellow', linewidth=2)
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, marker='>', color='purple', linewidth=2)
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, marker='>', color='orange', linewidth=2)

plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-65.png")
plt.imshow(img, zorder=0)
plt.show()
