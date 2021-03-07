import numpy as np
import matplotlib.pyplot as plt

### K=5 ZARA1 output at 40th second ###

# 0.9 speed
xSpeed01 = np.asarray([381, 406, 439, 474, 508, 541, 572, 604])
ySpeed01 = np.asarray([397, 399, 401, 401, 401, 401, 401, 402])

# GT speed
xSpeedGT = np.asarray([385, 408, 432, 457, 481, 506, 530, 554])
ySpeedGT = np.asarray([393, 390, 388, 388, 387, 387, 386, 385])

# 0.9 speed
#xSpeed02 = np.asarray([368, 391, 422, 456, 491, 524, 556, 587])
#ySpeed02 = np.asarray([424, 426, 426, 426, 425, 425, 425, 425])

xSpeed02 = np.asarray([361, 370, 375, 380, 384, 388, 391, 394])
ySpeed02 = np.asarray([423, 422, 421, 421, 420, 420, 420, 420])

# GT speed for 2nd ped
xSpeedGT02 = np.asarray([374, 398, 422, 446, 470, 495, 520, 545])
ySpeedGT02 = np.asarray([422, 421, 420, 419, 418, 418, 418, 417])

# 0.4 speed for 3rd ped
xSpeed03 = np.asarray([379, 395, 411, 427, 442, 457, 472, 487])
ySpeed03 = np.asarray([441, 441, 442, 442, 442, 442, 441, 440])

# GT speed for 2nd ped
xSpeedGT03 = np.asarray([386, 410, 434, 457, 481, 504, 528, 551])
ySpeedGT03 = np.asarray([439, 439, 439, 439, 439, 438, 437, 436])


plt.plot(xSpeed01,ySpeed01,zorder=1, color='purple', label='0.9', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='orange', label='0.1', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='red', label='0.4', linewidth=3, linestyle='--')
plt.plot(xSpeedGT,ySpeedGT,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')

plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='>', color='purple', linewidth=3)
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='>', color='orange', linewidth=3)
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, marker='>', color='red', linewidth=3)
plt.plot(xSpeedGT[-1],ySpeedGT[-1],zorder=1, marker='>', color='blue', linewidth=3)
plt.plot(xSpeedGT02[-1],ySpeedGT02[-1],zorder=1, marker='>', color='blue', linewidth=3)
plt.plot(xSpeedGT03[-1],ySpeedGT03[-1],zorder=1, marker='>', color='blue', linewidth=3)

plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-157.png")
plt.imshow(img, zorder=0)
plt.show()
