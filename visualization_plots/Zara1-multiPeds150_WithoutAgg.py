import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# Ped1
xSpeed01 = np.asarray([405, 383, 362, 340, 319, 298, 278, 258, 238, 218, 198, 178])
ySpeed01 = np.asarray([384, 383, 382, 380, 378, 376, 373, 371, 368, 366, 364, 362])

# Ped2
xSpeed02 = np.asarray([395, 373, 352, 330, 309, 289, 268, 248, 228, 209, 189, 169])
ySpeed02 = np.asarray([409, 406, 403, 400, 396, 393, 389, 386, 383, 380, 377, 373])

# Ped3
xSpeed03 = np.asarray([324, 348, 373, 398, 423, 448, 474, 500, 525, 551, 576, 601])
ySpeed03 = np.asarray([329, 332, 335, 337, 340, 342, 344, 345, 347, 348, 349, 351])

# Ped4
xSpeed04 = np.asarray([219, 244, 270, 295, 321, 348, 347, 401, 427, 454, 481, 507])
ySpeed04 = np.asarray([416, 423, 430, 436, 441, 446, 451, 455, 459, 462, 465, 469])

# Ped5
xSpeed05 = np.asarray([453, 435, 418, 401, 385, 369, 353, 338, 323, 308, 293, 278])
ySpeed05 = np.asarray([315, 310, 306, 301, 297, 293, 289, 285, 281, 278, 274, 271])

# Ped6
xSpeed06 = np.asarray([478, 457, 436, 416, 396, 376, 357, 337, 318, 300, 281, 262])
ySpeed06 = np.asarray([347, 343, 339, 335, 331, 327, 323, 320, 316, 312, 309, 305])

plt.plot(xSpeed01,ySpeed01,zorder=1, color='cyan', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='yellow', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='orange', linewidth=3, linestyle='--')
plt.plot(xSpeed05,ySpeed05,zorder=1, color='magenta', linewidth=3, linestyle='--')
plt.plot(xSpeed06,ySpeed06,zorder=1, color='purple', linewidth=3, linestyle='--')


plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, color='cyan', marker='<', linewidth=2, linestyle='--')
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, color='yellow', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, color='red', marker='>', linewidth=3, linestyle='--')
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, color='orange', marker='>', linewidth=3, linestyle='--')
plt.plot(xSpeed05[-1],ySpeed05[-1],zorder=1, color='magenta', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed06[-1],ySpeed06[-1],zorder=1, color='purple', marker='<', linewidth=3, linestyle='--')

#plt.legend()
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara150.png")
plt.imshow(img, zorder=0)
plt.show()
