import numpy as np
import matplotlib.pyplot as plt
#[63 63 64 64 65 65 65 65 66 66 66 66]
#[452 451 451 451 451 451 451 451 451 451 452 452]

xSpeed01 = np.asarray([487, 472, 457, 443, 429, 416, 402, 389, 376, 363, 351, 338])
ySpeed01 = np.asarray([403, 400, 398, 397, 395, 394, 393, 392, 391, 390, 390, 389])

xSpeed02 = np.asarray([512, 500, 488, 476, 465, 453, 442, 431, 420, 409, 398, 388])
ySpeed02 = np.asarray([343, 344, 344, 345, 345, 345, 345, 346, 346, 346, 347, 347])

xSpeed03 = np.asarray([503, 488, 473, 459, 445, 431, 418, 404, 391, 378, 366, 353])
ySpeed03 = np.asarray([367, 366, 365, 364, 364, 363, 362, 362, 362, 361, 361, 361])

xSpeed04 = np.asarray([313, 335, 357, 379, 400, 422, 444, 465, 487, 508, 530, 551])
ySpeed04 = np.asarray([424, 426, 428, 430, 432, 434, 436, 439, 441, 443, 446, 448])

xSpeed05 = np.asarray([257, 279, 302, 325, 347, 369, 392, 413, 435, 457, 478, 500])
ySpeed05 = np.asarray([393, 392, 392, 393, 394, 395, 396, 398, 400, 401, 403, 406])

xSpeed06 = np.asarray([572, 560, 547, 533, 520, 507, 493, 480, 467, 455, 442, 430])
ySpeed06 = np.asarray([324, 329, 333, 337, 340, 343, 346, 348, 351, 353, 356, 358])

xSpeed07 = np.asarray([573, 555, 535, 516, 497, 479, 461, 443, 425, 407, 390, 372])
ySpeed07 = np.asarray([356, 360, 363, 366, 369, 371, 373, 375, 376, 378, 380, 382])

#xSpeed08 = np.asarray()
#ySpeed08 = np.asarray()

plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='orange')
plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='purple')
plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='brown')
plt.plot(xSpeed05, ySpeed05, zorder=1, linewidth=3, linestyle='--', color='yellow')

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='orange')
plt.plot(xSpeed03[-1], ySpeed03[-1], zorder=1, marker='<', color='purple')
plt.plot(xSpeed04[-1], ySpeed04[-1], zorder=1, marker='>', color='brown')
plt.plot(xSpeed05[-1], ySpeed05[-1], zorder=1, marker='>', color='yellow')

#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-208.png")
plt.imshow(img, zorder=0)
plt.show()
