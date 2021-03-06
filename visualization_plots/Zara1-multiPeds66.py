import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# Ped1
xSpeed01 = np.asarray([85, 82, 80, 79, 79, 79, 80, 80])
ySpeed01 = np.asarray([282, 283, 284, 285, 286, 288, 289, 290])

# Ped2
xSpeed02 = np.asarray([248, 273, 297, 322, 347, 373, 398])
ySpeed02 = np.asarray([372, 372, 373, 374, 376, 378, 380])

# Ped3
xSpeed03 = np.asarray([246, 270, 295, 320, 345, 371, 397])
ySpeed03 = np.asarray([343, 343, 345, 347, 350, 352, 355])

# Ped4
xSpeed04 = np.asarray([444, 418, 391, 365, 339, 313, 287])
ySpeed04 = np.asarray([379, 380, 380, 381, 382, 382, 383])

# Ped5
xSpeed05 = np.asarray([450, 423, 397, 370, 344, 318, 292])
ySpeed05 = np.asarray([411, 411, 411, 411, 411, 411, 411])

# 0.9 speed
#xSpeed02 = np.asarray([255, 235, 215, 195, 169, 140, 109, 78, 47, 16, -15, -46])
#ySpeed02 = np.asarray([353, 350, 348, 347, 346, 346, 346, 346, 346, 347, 347, 347])

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
