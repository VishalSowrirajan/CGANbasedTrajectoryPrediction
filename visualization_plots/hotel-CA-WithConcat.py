import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3

xSpeed01 = np.asarray([164, 163, 162, 161, 160, 159, 158, 157])
ySpeed01 = np.asarray([234, 234, 234, 235, 235, 236, 238, 239])

xSpeed02 = np.asarray([444, 443, 441, 439, 437, 435, 433, 432])
ySpeed02 = np.asarray([224, 224, 223, 223, 223, 224, 225, 227])

xSpeed03 = np.asarray([328, 300, 274, 250, 226, 201, 177, 154])
ySpeed03 = np.asarray([363, 372, 382, 392, 402, 411, 420, 430])

xSpeed04 = np.asarray([324, 295, 263, 229, 193, 156, 120,  88])
ySpeed04 = np.asarray([311, 316, 324, 333, 344, 357, 372, 388])

xSpeed05 = np.asarray([293, 317, 341, 365, 386, 407, 426, 444])
ySpeed05 = np.asarray([391, 386, 383, 382, 380, 378, 373, 367])

xSpeed06 = np.asarray([287, 312, 336, 360, 382, 403, 424, 442])
ySpeed06 = np.asarray([419, 415, 413, 412, 411, 409, 406, 401])

ySpeed07 = np.asarray([475, 447, 416, 388, 363, 339, 313, 287])
xSpeed07 = np.asarray([352, 350, 346, 345, 344, 342, 339, 337])


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
