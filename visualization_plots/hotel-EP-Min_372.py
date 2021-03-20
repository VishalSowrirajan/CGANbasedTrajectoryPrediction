import numpy as np
import matplotlib.pyplot as plt


#xSpeed01 = np.asarray([314, 283, 254, 228, 202, 177, 154, 131])
#ySpeed01 = np.asarray([326, 334, 345, 356, 367, 378, 390, 402])

#xSpeed02 = np.asarray([321, 290, 261, 234, 209, 184, 160, 137])
#ySpeed02 = np.asarray([354, 362, 372, 383, 394, 405, 417, 429])

xSpeed01 = np.asarray([312, 281, 253, 227, 203, 179, 156, 134])
ySpeed01 = np.asarray([325, 332, 342, 352, 362, 373, 384, 396])

xSpeed02 = np.asarray([319, 288, 260, 234, 210, 186, 163, 141])
ySpeed02 = np.asarray([353, 359, 369, 379, 389, 400, 411, 423])


xSpeedGT = np.asarray([307, 269, 231, 194, 156, 114,  77,  41])
ySpeedGT = np.asarray([330, 332, 334, 333, 332, 332, 334, 342])

xSpeedGT02 = np.asarray([313, 276, 239, 201, 162, 121,  88,  46])
ySpeedGT02 = np.asarray([358, 358, 361, 361, 359, 363, 364, 372])


plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red', label='0.9')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='red')

plt.plot(xSpeedGT, ySpeedGT, zorder=1, linewidth=3, color='blue', label='GT', linestyle='--')
plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, linewidth=3, color='blue', linestyle='--')#

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='red')

plt.plot(xSpeedGT[-1], ySpeedGT[-1], zorder=1, marker='<', color='blue')
plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='<', color='blue')


#plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel500.png")
plt.imshow(img, zorder=0)
plt.show()
