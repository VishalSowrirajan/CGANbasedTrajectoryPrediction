import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# 0.9 speed
#xSpeed01 = np.asarray([255, 235, 215, 195, 169, 140, 109, 78, 47, 16, -15, -46])
#ySpeed01 = np.asarray([353, 350, 348, 347, 346, 346, 346, 346, 346, 347, 347, 347])
xSpeed01 = np.asarray([255, 235, 215, 195, 169, 140, 109, 78, 47, 16])
ySpeed01 = np.asarray([353, 350, 348, 347, 346, 346, 346, 346, 346, 347])

# GT speed
xSpeedGT = np.asarray([255, 236, 215, 192, 169, 146, 124, 104, 84, 65])
ySpeedGT = np.asarray([352, 347, 342, 341, 340, 339, 338, 335, 331, 329])

# Stopping after 4th frame
xSpeed03 = np.asarray([255, 234, 215, 195, 188, 189, 191, 192, 193])
ySpeed03 = np.asarray([353, 350, 347, 346, 345, 345, 344, 344, 343])


plt.plot(xSpeed01,ySpeed01,zorder=1, color='red', label='0.9', linewidth=3, linestyle='--')
plt.plot(xSpeedGT,ySpeedGT,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='yellow', label='0.0', linewidth=3, linestyle='--')

plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='<', color='red', linewidth=3)
plt.plot(xSpeedGT[-1],ySpeedGT[-1],zorder=1, marker='<', color='blue', linewidth=3)
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, marker='<', color='yellow', linewidth=3)

#plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='>', color='red', linewidth=2)
#plt.plot(xSpeedGT[-1],ySpeedGT[-1],zorder=1, marker='>', color='blue', linewidth=2)

plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-19.png")
plt.imshow(img, zorder=0)
plt.show()
