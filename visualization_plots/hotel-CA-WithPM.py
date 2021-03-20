import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3

xSpeed01 = np.asarray([165, 165, 166, 166, 167, 168, 169, 170])
ySpeed01 = np.asarray([235, 236, 236, 237, 237, 238, 238, 238])

xSpeed02 = np.asarray([445, 445, 444, 444, 444, 445, 445, 446])
ySpeed02 = np.asarray([227, 228, 229, 231, 231, 232, 232, 232])

xSpeed03 = np.asarray([330, 304, 276, 249, 221, 194, 168, 142])
ySpeed03 = np.asarray([357, 359, 361, 363, 365, 368, 371, 375])

xSpeed04 = np.asarray([325, 297, 268, 238, 207, 177, 148, 120])
ySpeed04 = np.asarray([305, 307, 309, 311, 313, 315, 317, 319])

xSpeed05 = np.asarray([297, 322, 345, 368, 389, 410, 431, 451])
ySpeed05 = np.asarray([393, 394, 399, 405, 412, 420, 427, 434])

xSpeed06 = np.asarray([292, 317, 341, 363, 385, 406, 426, 446])
ySpeed06 = np.asarray([422, 424, 430, 437, 445, 453, 460, 467])

xSpeed07 = np.asarray([354, 353, 349, 345, 341, 338, 335, 332])
ySpeed07 = np.asarray([473, 446, 418, 391, 363, 335, 308, 281])


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
