import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3


xSpeed01 = np.asarray([506, 483, 461, 440, 417, 394, 370, 345])
ySpeed01 = np.asarray([314, 313, 314, 314, 315, 316, 316, 315])

xSpeed02 = np.asarray([504, 481, 458, 431])
ySpeed02 = np.asarray([315, 317, 318, 320])

xSpeedGT = np.asarray([556, 540, 524, 508, 492, 473])
ySpeedGT = np.asarray([311, 312, 314, 323, 328, 332])

xSpeedGT02 = np.asarray([563, 547, 532, 516, 500, 483])
ySpeedGT02 = np.asarray([281, 281, 284, 284, 284, 287])


#plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red', label='0.6')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='red')

#plt.plot(xSpeedGT, ySpeedGT, zorder=1, linewidth=3, color='blue', label='GT')
#plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, linewidth=3, color='blue')

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='red')

#plt.plot(xSpeedGT[-1], ySpeedGT[-1], zorder=1, marker='<', color='blue')
#plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='<', color='blue')

plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel22.png")
plt.imshow(img, zorder=0)
plt.show()
