import numpy as np
import matplotlib.pyplot as plt

# GROUP AVOIDANCE PLOT - PLOTTING HOTEL DATASET - 114 second

xSpeed05 = np.asarray([263, 237, 216, 196, 178, 161, 145, 130])
ySpeed05 = np.asarray([384, 371, 360, 352, 347, 342, 338, 334])

xSpeed06 = np.asarray([265, 240, 218, 199, 180, 162, 146, 130])
ySpeed06 = np.asarray([344, 332, 323, 317, 313, 309, 307, 304])

xSpeed07 = np.asarray([262, 232, 203, 174, 144, 114,  83,  53])
ySpeed07 = np.asarray([393, 394, 399, 405, 412, 421, 429, 437])

xSpeed08 = np.asarray([264, 234, 205, 176, 146, 116,  86,  55])
ySpeed08 = np.asarray([351, 353, 358, 364, 372,381 ,389 ,397])

xSpeedGT01 = np.asarray([254, 224, 191, 158, 126, 93, 60, 28])
ySpeedGT01 = np.asarray([393, 399, 404, 413, 421, 425, 430, 430])

xSpeedGT02 = np.asarray([257, 223, 188, 154, 121, 81,  46,  16])
ySpeedGT02 = np.asarray([349, 345, 342, 335, 334, 335, 335, 345])

plt.plot(xSpeed05, ySpeed05, zorder=1, color='red', linewidth=3, linestyle='--', label='0.6')
#plt.plot(xSpeed06, ySpeed06, zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed07, ySpeed07, zorder=1, color='purple', linewidth=3, linestyle='--', label='0.8')
#plt.plot(xSpeed08, ySpeed08, zorder=1, color='purple', linewidth=3, linestyle='--')

plt.plot(xSpeedGT01, ySpeedGT01, zorder=1, color='blue', linewidth=3, linestyle='--', label='GT')
#plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, color='blue', linewidth=3, linestyle='--')

plt.plot(xSpeed05[-1], ySpeed05[-1], zorder=1, color='red', marker='<')
plt.plot(xSpeed07[-1], ySpeed07[-1], zorder=1, color='purple', marker='<')
plt.plot(xSpeedGT01[-1], ySpeedGT01[-1], zorder=1, color='blue', marker='<')
#plt.plot(xSpeed06[-1], ySpeed06[-1], zorder=1, color='red', marker='<')
img = plt.imread("../hotel115.png")

plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
plt.imshow(img, zorder=0)
plt.show()
