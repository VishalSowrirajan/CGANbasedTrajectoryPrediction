import numpy as np
import matplotlib.pyplot as plt
#[64, 65, 66, 67, 68, 69, 70, 71]
#[453, 453, 454, 455, 456, 456, 457, 459]

xSpeed01 = np.asarray([487, 473, 458, 443, 429, 414, 400, 385])
ySpeed01 = np.asarray([403, 401, 398, 396, 394, 392, 391, 390])

xSpeed02 = np.asarray([510, 495, 481, 467, 453, 439, 425, 411])
ySpeed02 = np.asarray([342, 342, 340, 339, 338, 336, 336, 335])

xSpeed03 = np.asarray([503, 488, 472, 457, 441, 426, 411, 396])
ySpeed03 = np.asarray([366, 365, 364, 363, 362, 362, 361, 361])

xSpeed04 = np.asarray([315, 337, 359, 382, 405, 427, 449, 471])
ySpeed04 = np.asarray([427, 431, 435, 441, 447, 454, 461, 469])

xSpeed05 = np.asarray([258, 281, 304, 327, 351, 373, 396, 418])
ySpeed05 = np.asarray([395, 395, 397, 400, 403, 408, 413, 419])

xSpeed06 = np.asarray([574, 561, 546, 530, 515, 499, 484, 468])
ySpeed06 = np.asarray([325, 330, 333, 336, 338, 341, 343, 344])

xSpeed07 = np.asarray([576, 558, 538, 518, 498, 478, 458, 439])
ySpeed07 = np.asarray([357, 361, 364, 367, 369, 371, 373, 375])

plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='orange')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='orange')
plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='orange')
plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='red')
plt.plot(xSpeed05, ySpeed05, zorder=1, linewidth=3, linestyle='--', color='yellow')
plt.plot(xSpeed06, ySpeed06, zorder=1, linewidth=3, linestyle='--', color='blue')
plt.plot(xSpeed07, ySpeed07, zorder=1, linewidth=3, linestyle='--', color='blue')

plt.plot(xSpeed04[-1], ySpeed04[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed03[-1], ySpeed03[-1], zorder=1, marker='<', color='orange')
plt.plot(xSpeed05[-1], ySpeed05[-1], zorder=1, marker='>', color='yellow')
plt.plot(xSpeed06[-1], ySpeed06[-1], zorder=1, marker='>', color='purple')

#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-208.png")
plt.imshow(img, zorder=0)
plt.show()
