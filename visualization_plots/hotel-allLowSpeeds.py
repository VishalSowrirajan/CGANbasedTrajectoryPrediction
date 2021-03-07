import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3

# PEDS WITH 0.8 speed
xSpeed01 = np.asarray([377, 405, 432, 459, 486, 513, 540, 568])
ySpeed01 = np.asarray([353, 357, 359, 358, 356, 354, 352, 351])

xSpeed02 = np.asarray([380, 409, 437, 464, 492, 521, 550, 579])
ySpeed02 = np.asarray([403, 412, 419, 423, 427, 431, 434, 437])

xSpeed03 = np.asarray([399, 428, 456, 484, 512, 541, 570, 600])
ySpeed03 = np.asarray([458, 467, 472, 476, 478, 480, 482, 484])


# PEDS WITH 0.5 speed
xSpeed04 = np.asarray([376, 403, 428, 451, 473, 494, 516, 537])
ySpeed04 = np.asarray([355, 359, 360, 359, 358, 357, 356, 356])

xSpeed05 = np.asarray([379, 407, 432, 456, 479, 502, 524, 547])
ySpeed05 = np.asarray([404, 413, 417, 419, 419, 420, 420, 421])

xSpeed06 = np.asarray([398, 426, 451, 475, 498, 521, 544, 566])
ySpeed06 = np.asarray([460, 467, 471, 473, 473, 473, 473, 474])

# PEDS GT
xSpeedGT01 = np.asarray([385, 418, 451, 484, 516, 546, 578, 605])
ySpeedGT01 = np.asarray([343, 339, 341, 342, 340, 339, 331, 325])

xSpeedGT02 = np.asarray([388, 424, 457, 490, 521, 552, 584, 616])
ySpeedGT02 = np.asarray([393, 383, 379, 372, 370, 369, 361, 361])

xSpeedGT03 = np.asarray([405, 437, 478, 511, 543, 573, 607, 638])
ySpeedGT03 = np.asarray([446, 442, 435, 423, 420, 419, 422, 430])

plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red', label='0.8')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='red')
#plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='red')

plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='purple', label='0.5')
plt.plot(xSpeed05, ySpeed05, zorder=1, linewidth=3, linestyle='--', color='purple')
#plt.plot(xSpeed06, ySpeed06, zorder=1, linewidth=3, linestyle='--', color='purple')

plt.plot(xSpeedGT01, ySpeedGT01, zorder=1, linewidth=3, linestyle='--', color='blue', label='GT')
plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, linewidth=3, linestyle='--', color='blue')
#plt.plot(xSpeedGT03, ySpeedGT03, zorder=1, linewidth=3, linestyle='--', color='blue')

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='>', color='red')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='>', color='red')
#plt.plot(xSpeed03[-1], ySpeed03[-1], zorder=1, marker='>', color='red')

plt.plot(xSpeed04[-1], ySpeed04[-1], zorder=1, marker='>', color='purple')
plt.plot(xSpeed05[-1], ySpeed05[-1], zorder=1, marker='>', color='purple')
#plt.plot(xSpeed06[-1], ySpeed06[-1], zorder=1, marker='>', color='purple')

plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='>', color='blue')
plt.plot(xSpeedGT01[-1], ySpeedGT01[-1], zorder=1, marker='>', color='blue')

plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel302.png")
plt.imshow(img, zorder=0)
plt.show()
