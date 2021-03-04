import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# Ped1
xSpeed01 = np.asarray([85, 82, 79, 77, 75, 73, 71, 68, 66, 63, 59, 55])
ySpeed01 = np.asarray([283, 285, 287, 289, 291, 293, 295, 296, 298, 299, 300, 301])

# Ped2
xSpeed02 = np.asarray([248, 273, 299, 326, 354, 382, 410, 438, 466, 494, 522, 551])
ySpeed02 = np.asarray([374, 373, 373, 373, 373, 373, 372, 372, 372, 371, 371, 370])

# Ped3
xSpeed03 = np.asarray([246, 272, 299, 326, 354, 382, 411, 439, 468, 496, 525, 553])
ySpeed03 = np.asarray([344, 344, 345, 346, 347, 348, 349, 350, 350, 351, 351, 351])

# Ped4
xSpeed04 = np.asarray([442, 413, 383, 353, 323, 292, 261, 230, 199, 168, 138, 107])
ySpeed04 = np.asarray([383, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406])

# Ped5
xSpeed05 = np.asarray([448, 418, 388, 357, 326, 295, 263, 232, 200, 169, 138, 107])
ySpeed05 = np.asarray([415, 418, 421, 423, 425, 427, 429, 431, 433, 435, 436, 439])

# 0.9 speed
#xSpeed02 = np.asarray([255, 235, 215, 195, 169, 140, 109, 78, 47, 16, -15, -46])
#ySpeed02 = np.asarray([353, 350, 348, 347, 346, 346, 346, 346, 346, 347, 347, 347])

# GT speed
xSpeedGT = np.asarray([255, 236, 215, 192, 169, 146, 124, 104, 84, 65, 50, 35])
ySpeedGT = np.asarray([352, 347, 342, 341, 340, 339, 338, 335, 331, 329, 330, 332])


plt.plot(xSpeed01,ySpeed01,zorder=1, color='green', linewidth=2, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='yellow', linewidth=2, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='red', linewidth=2, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='orange', linewidth=2, linestyle='--')
plt.plot(xSpeed05,ySpeed05,zorder=1, color='pink', linewidth=2, linestyle='--')

#plt.plot(xSpeedGT,ySpeedGT,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeed02,ySpeed02,zorder=1, color='yellow', label='0.9', linewidth=3, linestyle='--')

plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, color='green', marker='<', linewidth=2, linestyle='--')
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, color='yellow', marker='>', linewidth=2, linestyle='--')
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, color='red', marker='>', linewidth=2, linestyle='--')
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, color='orange', marker='<', linewidth=2, linestyle='--')
plt.plot(xSpeed05[-1],ySpeed05[-1],zorder=1, color='pink', marker='<', linewidth=2, linestyle='--')

#plt.legend()
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-66.png")
plt.imshow(img, zorder=0)
plt.show()
