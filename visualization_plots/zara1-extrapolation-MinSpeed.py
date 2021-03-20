import numpy as np
import matplotlib.pyplot as plt

xSpeed01 = np.asarray([330, 339, 346, 352, 357, 361, 364, 366, 368, 369, 370, 371])
ySpeed01 = np.asarray([368, 371, 373, 376, 378, 380, 381, 383, 385, 386, 387, 388])

xSpeed02 = np.asarray([315, 328, 337, 344, 350, 354, 357, 360, 362, 363, 365, 366])
ySpeed02 = np.asarray([338, 338, 339, 339, 340, 340, 340, 340, 340, 340, 340, 340])

#GT
xSpeed03 = np.asarray([328, 346, 364, 382, 404, 425, 447, 468, 489, 510, 532, 554])
ySpeed03 = np.asarray([369, 373, 376, 380, 383, 387, 390, 392, 394, 396, 396, 396])

xSpeed04 = np.asarray([319, 339, 360, 381, 402, 423, 444, 465, 486, 507, 526, 546])
ySpeed04 = np.asarray([341, 345, 347, 348, 350, 352, 354, 355, 357, 360, 364, 367])


plt.plot(xSpeed03,ySpeed03,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
plt.plot(xSpeed01,ySpeed01,zorder=1, color='red', label='0.0', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='blue', linewidth=3, linestyle='--')

plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='>', color='red', linewidth=2)
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='>', color='red', linewidth=2)
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, marker='>', color='blue', linewidth=2)
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, marker='>', color='blue', linewidth=2)

plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara80.png")
plt.imshow(img, zorder=0)
plt.show()
