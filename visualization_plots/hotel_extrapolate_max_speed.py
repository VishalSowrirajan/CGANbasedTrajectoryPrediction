import numpy as np
import matplotlib.pyplot as plt


xSpeed01 = np.asarray([504, 484, 460, 435, 408, 379, 348, 316, 283, 248, 212, 174])
ySpeed01 = np.asarray([316, 318, 320, 322, 324, 325, 325, 324, 322, 319, 314, 307])

ySpeed02 = np.asarray([286, 289, 291, 294, 296, 299, 300, 301, 301, 301, 299, 296])
xSpeed02 = np.asarray([513, 493, 470, 446, 419, 391, 362, 331, 298, 265, 230, 194])

xSpeedGT = np.asarray([508, 492, 473, 455, 436, 419, 402, 384, 366, 346, 322, 299])
ySpeedGT = np.asarray([323, 328, 332, 338, 343, 346, 352, 357, 359, 362, 369, 371])

xSpeedGT02 = np.asarray([516, 500, 483, 464, 444, 425, 406, 388, 368, 349, 331, 312])
ySpeedGT02 = np.asarray([284, 284, 287, 286, 292, 295, 298, 304, 307, 314, 318, 324])


plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red', label='0.9')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='red')

#plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='cyan', label='0.5')
#plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='cyan')


plt.plot(xSpeedGT, ySpeedGT, zorder=1, linewidth=3, color='blue', label='GT', linestyle='--')
plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, linewidth=3, color='blue', linestyle='--')

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='red')

plt.plot(xSpeedGT[-1], ySpeedGT[-1], zorder=1, marker='<', color='blue')
plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='<', color='blue')


#plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel22.png")
plt.imshow(img, zorder=0)
plt.show()
