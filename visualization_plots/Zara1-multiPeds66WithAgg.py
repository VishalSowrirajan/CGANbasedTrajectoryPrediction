import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# Ped1
xSpeed01 = np.asarray([83, 78, 76, 75, 74, 73, 72, 72, 71, 70, 68, 66])
ySpeed01 = np.asarray([282, 284, 286, 288, 290, 293, 296, 299, 303, 306, 310, 313])

# Ped2
xSpeed02 = np.asarray([247, 272, 298, 324, 348, 372, 395, 418, 442, 466, 490, 514])
ySpeed02 = np.asarray([373, 373, 374, 375, 376, 375, 374, 373, 372, 370, 369, 367])

# Ped3
xSpeed03 = np.asarray([244, 270, 296, 322, 348, 371, 395, 419, 444, 468, 493, 518])
ySpeed03 = np.asarray([343, 344, 346, 348, 348, 348, 348, 348, 347, 346, 345, 344])

# Ped4
xSpeed04 = np.asarray([445, 417, 388, 357, 326, 296, 267, 239, 210, 182, 154, 125])
ySpeed04 = np.asarray([382, 386, 390, 393, 394, 394, 396, 399, 401, 404, 406, 408])

# Ped5
xSpeed05 = np.asarray([454, 429, 401, 373, 344, 315, 288, 261, 234, 207, 180, 154])
ySpeed05 = np.asarray([414, 417, 421, 423, 424, 425, 427, 429, 431, 433, 434, 435])


# GT speed
xSpeedGT = np.asarray([255, 236, 215, 192, 169, 146, 124, 104, 84, 65, 50, 35])
ySpeedGT = np.asarray([352, 347, 342, 341, 340, 339, 338, 335, 331, 329, 330, 332])


#plt.plot(xSpeed01,ySpeed01,zorder=1, color='green', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='yellow', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='orange', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed05,ySpeed05,zorder=1, color='purple', linewidth=3, linestyle='--')

#plt.plot(xSpeedGT,ySpeedGT,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeed02,ySpeed02,zorder=1, color='yellow', label='0.9', linewidth=3, linestyle='--')

#plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, color='green', marker='<', linewidth=2, linestyle='--')
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, color='yellow', marker='>', linewidth=3, linestyle='--')
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, color='orange', marker='>', linewidth=3, linestyle='--')
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed05[-1],ySpeed05[-1],zorder=1, color='purple', marker='<', linewidth=3, linestyle='--')

#plt.legend()
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara65.png")
plt.imshow(img, zorder=0)
plt.show()
