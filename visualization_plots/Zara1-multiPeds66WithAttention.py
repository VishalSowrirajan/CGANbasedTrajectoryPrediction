import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# Ped1
xSpeed01 = np.asarray([86, 84, 82, 79, 77, 74, 72, 71, 69, 68, 66, 55])
ySpeed01 = np.asarray([284, 285, 285, 286, 286, 287, 288, 288, 289, 289, 289, 289])

# Ped2
xSpeed02 = np.asarray([247, 272, 296, 319, 343, 366, 391, 415, 439, 463, 487, 511])
ySpeed02 = np.asarray([374, 374, 372, 370, 368, 366, 364, 363, 361, 360, 359, 358])

# Ped3
xSpeed03 = np.asarray([245, 270, 294, 318, 342, 366, 390, 415, 439, 464, 488, 512])
ySpeed03 = np.asarray([344, 345, 344, 343, 341, 340, 339, 338, 337, 337, 336, 335])

# Ped4
xSpeed04 = np.asarray([446, 421, 394, 367, 340, 314, 289, 264, 239, 215, 191, 168])
ySpeed04 = np.asarray([383, 384, 385, 387, 388, 390, 391, 394, 396, 399, 402, 405])

# Ped5
xSpeed05 = np.asarray([452, 426, 399, 372, 345, 318, 293, 267, 243, 218, 194, 170])
ySpeed05 = np.asarray([415, 415, 417, 418, 419, 420, 422, 424, 426, 428, 431, 434])

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
