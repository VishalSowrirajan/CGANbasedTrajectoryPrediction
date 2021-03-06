import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# Ped1
xSpeed01 = np.asarray([404, 381, 360, 338, 316, 294, 273, 251, 229, 207, 185, 163])
ySpeed01 = np.asarray([384, 382, 381, 379, 378, 377, 375, 374, 373, 372, 370, 369])

# Ped2
xSpeed02 = np.asarray([394, 372, 350, 329, 307, 286, 265, 244, 223, 202, 181, 160])
ySpeed02 = np.asarray([410, 407, 404, 401, 398, 395, 392, 389, 386, 384, 381, 378])

# Ped3
xSpeed03 = np.asarray([323, 347, 372, 397, 423, 448, 474, 500, 526, 552, 578, 605])
ySpeed03 = np.asarray([331, 336, 342, 347, 353, 359, 365, 370, 375, 380, 384, 389])

# Ped4
xSpeed04 = np.asarray([217, 240, 265, 292, 319, 347, 376, 405, 434, 464, 494, 524])
ySpeed04 = np.asarray([417, 426, 434, 442, 449, 456, 463, 470, 476, 482, 488, 493])

# Ped5
xSpeed05 = np.asarray([449, 429, 410, 391, 373, 354, 337, 319, 301, 283, 266, 248])
ySpeed05 = np.asarray([316, 311, 306, 302, 298, 295, 291, 288, 285, 281, 278, 275])

# Ped6
xSpeed06 = np.asarray([476, 453, 432, 410, 389, 367, 346, 325, 303, 282, 261, 239])
ySpeed06 = np.asarray([348, 343, 339, 336, 332, 329, 326, 322, 319, 316, 313, 310])

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
