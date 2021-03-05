import numpy as np
import matplotlib.pyplot as plt

### K=5 ZARA1 output at 40th second ###

# 0.2 speed
xSpeed01 = np.asarray([191, 202, 209, 214, 219, 224, 230, 236, 241, 247, 253, 259])
ySpeed01 = np.asarray([310, 308, 307, 306, 307, 308, 309, 311, 312, 314, 316, 318])

# 0.4 speed
xSpeed02 = np.asarray([193, 208, 222, 236, 251, 266, 281, 297, 312, 327, 342, 357])
ySpeed02 = np.asarray([311, 310, 310, 311, 311, 312, 313, 313, 314, 314, 314, 314])

# 0.6 speed
xSpeed03 = np.asarray([196, 218, 241, 265, 288, 312, 355, 359, 382, 406, 429, 452])
ySpeed03 = np.asarray([313, 316, 319, 321, 324, 326, 327, 328, 329, 330, 331, 332])

# 0.8 speed
xSpeed04 = np.asarray([199, 226, 256, 288, 320, 352, 383, 414, 444, 474, 505, 535])
ySpeed04 = np.asarray([313, 315, 318, 321, 323, 324, 325, 326, 326, 327, 327, 328])

# GT speed
xSpeedGT = np.asarray([198, 221, 243, 266, 288, 310, 332, 354, 375, 399, 422, 446])
ySpeedGT = np.asarray([313, 316, 319, 322, 326, 328, 328, 329, 329, 331, 333, 334])


plt.plot(xSpeed01,ySpeed01,zorder=1, color='yellow', label='0.2', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='green', label='0.4', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='brown', label='0.6', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='red', label='0.8', linewidth=3, linestyle='--')
plt.plot(xSpeedGT,ySpeedGT,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')

plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='>', color='yellow', linewidth=3)
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='>', color='green', linewidth=3)
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, marker='>', color='brown', linewidth=3)
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, marker='>', color='red', linewidth=3)
plt.plot(xSpeedGT[-1],ySpeedGT[-1],zorder=1, marker='>', color='blue', linewidth=3)

#plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../8th second.png")
plt.imshow(img, zorder=0)
plt.show()
