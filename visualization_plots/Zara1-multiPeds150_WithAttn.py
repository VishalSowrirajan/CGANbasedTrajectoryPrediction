import numpy as np
import matplotlib.pyplot as plt

### Zara1 Stop Ped 20th second ###

# Ped1
xSpeed01 = np.asarray([405, 384, 362, 341, 319, 298, 278, 258, 238, 219, 200, 182])
ySpeed01 = np.asarray([384, 384, 384, 384, 385, 385, 385, 386, 387, 388, 390, 391])

# Ped2
xSpeed02 = np.asarray([395, 374, 352, 330, 308, 287, 267, 246, 227, 207, 188, 169])
ySpeed02 = np.asarray([408, 406, 404, 402, 401, 399, 398, 397, 396, 395, 395, 394])

# Ped3
xSpeed03 = np.asarray([324, 350, 375, 400, 424, 449, 473, 498, 522, 546, 571, 595])
ySpeed03 = np.asarray([328, 330, 331, 331, 332, 332, 333, 334, 335, 336, 337, 338])

# Ped4
xSpeed04 = np.asarray([219, 244, 272, 299, 327, 354, 382, 409, 436, 463, 490, 518])
ySpeed04 = np.asarray([415, 422, 427, 432, 436, 440, 443, 447, 450, 453, 456, 459])

# Ped5
xSpeed05 = np.asarray([453, 437, 422, 406, 390, 373, 357, 341, 325, 309, 294, 279])
ySpeed05 = np.asarray([313, 307, 300, 295, 290, 285, 281, 277, 274, 270, 267, 264])

# Ped6
xSpeed06 = np.asarray([478, 458, 438, 417, 396, 376, 356, 337, 318, 299, 281, 263])
ySpeed06 = np.asarray([346, 341, 337, 333, 330, 327, 324, 321, 319, 316, 314, 312])

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
