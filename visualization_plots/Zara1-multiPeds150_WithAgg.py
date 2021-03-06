import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# Ped1
xSpeed01 = np.asarray([403, 380, 355, 331, 307, 285, 263, 241, 219, 198, 176, 155])
ySpeed01 = np.asarray([385, 385, 385, 385, 386, 387, 388, 389, 389, 389, 389, 388])

# Ped2
xSpeed02 = np.asarray([393, 370, 346, 323, 301, 279, 258, 237, 216, 196, 175, 155])
ySpeed02 = np.asarray([409, 406, 404, 403, 402, 401, 400, 399, 397, 395, 393, 391])

# Ped3
xSpeed03 = np.asarray([322, 346, 372, 397, 421, 444, 468, 491, 515, 539, 563, 588])
ySpeed03 = np.asarray([328, 329, 332, 335, 336, 337, 337, 337, 336, 336, 335, 334])

# Ped4
xSpeed04 = np.asarray([216, 239, 265, 292, 318, 344, 371, 399, 426, 454, 482, 510])
ySpeed04 = np.asarray([416, 424, 432, 439, 444, 449, 454, 458, 463, 467, 472, 476])

# Ped5
xSpeed05 = np.asarray([448, 428, 409, 390, 371, 353, 335, 317, 299, 282, 264, 246])
ySpeed05 = np.asarray([315, 311, 308, 306, 304, 301, 299, 296, 292, 289, 285, 282])

# Ped6
xSpeed06 = np.asarray([475, 452, 428, 405, 383, 361, 339, 317, 295, 273, 251, 230])
ySpeed06 = np.asarray([347, 344, 342, 341, 340, 339, 337, 336, 334, 331, 329, 326])

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
