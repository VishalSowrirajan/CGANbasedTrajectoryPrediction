import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3

xSpeed01 = np.asarray([165, 165, 165, 165, 165, 166, 166, 166])
ySpeed01 = np.asarray([233, 232, 231, 231, 230, 229, 229, 228])

xSpeed02 = np.asarray([445, 444, 443, 443, 443, 443, 443, 443])
ySpeed02 = np.asarray([225, 225, 224, 223, 223, 222, 222, 222])

xSpeed03 = np.asarray([327, 301, 276, 252, 228, 203, 178, 153])
ySpeed03 = np.asarray([354, 351, 348, 346, 345, 344, 345, 346])

xSpeed04 = np.asarray([323, 295, 269, 243, 217, 191, 165, 139])
ySpeed04 = np.asarray([307, 311, 314, 318, 324, 330, 336, 343])

xSpeed05 = np.asarray([297, 322, 346, 367, 387, 406, 424, 442])
ySpeed05 = np.asarray([394, 394, 393, 391, 389, 386, 383, 379])

xSpeed06 = np.asarray([291, 317, 341, 363, 383, 402, 420, 437])
ySpeed06 = np.asarray([422, 423, 422, 421, 419, 416, 413, 410])

xSpeed07 = np.asarray([471, 445, 420, 396, 372, 347, 322, 298])
ySpeed07 = np.asarray([350, 345, 341, 337, 334, 332, 330, 330])


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
