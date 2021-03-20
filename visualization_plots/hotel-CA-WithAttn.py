import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3

xSpeed01 = np.asarray([165, 165, 165, 165, 165, 165, 165, 165])
ySpeed01 = np.asarray([233, 232, 231, 231, 231, 231, 231, 230])

xSpeed02 = np.asarray([444, 444, 443, 443, 442, 442, 442, 442])
ySpeed02 = np.asarray([225, 224, 224, 224, 224, 224, 224, 224])

xSpeed03 = np.asarray([328, 303, 279, 255, 231, 206, 181, 156])
ySpeed03 = np.asarray([356, 355, 354, 354, 355, 356, 359, 362])

xSpeed04 = np.asarray([324, 298, 273, 248, 222, 197, 171, 145])
ySpeed04 = np.asarray([310, 315, 321, 327, 335, 343, 351, 360])

xSpeed05 = np.asarray([297, 323, 347, 369, 389, 409, 427, 445])
ySpeed05 = np.asarray([395, 396, 397, 397, 398, 397, 396, 394])

ySpeed06 = np.asarray([423, 424, 426, 427, 428, 427, 427, 425])
xSpeed06 = np.asarray([292, 318, 342, 364, 385, 404, 423, 441])


#plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red', label='0.6')
#plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='green')
plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='orange')
plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='red')
plt.plot(xSpeed05, ySpeed05, zorder=1, linewidth=3, linestyle='--', color='yellow')
plt.plot(xSpeed06, ySpeed06, zorder=1, linewidth=3, linestyle='--', color='purple')
#plt.plot(xSpeed07, ySpeed07, zorder=1, linewidth=3, linestyle='--', color='violet')

plt.plot(xSpeed04[-1], ySpeed04[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed03[-1], ySpeed03[-1], zorder=1, marker='<', color='orange')
plt.plot(xSpeed05[-1], ySpeed05[-1], zorder=1, marker='>', color='yellow')
plt.plot(xSpeed06[-1], ySpeed06[-1], zorder=1, marker='>', color='purple')

#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel383.png")
plt.imshow(img, zorder=0)
plt.show()
