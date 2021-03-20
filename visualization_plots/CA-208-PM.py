import numpy as np
import matplotlib.pyplot as plt
#[66 67 69 70 71 71 71 72 72 71 71 71]
#[453 454 454 454 455 455 455 456 456 456 456 457]

xSpeed01 = np.asarray([489, 474, 459, 445, 431, 417, 404, 391, 378, 365, 352, 340])
ySpeed01 = np.asarray([404, 403, 402, 400, 399, 398, 396, 395, 394, 393, 392, 390])

xSpeed02 = np.asarray([513, 501, 488, 475, 462, 450, 438, 425, 413, 402, 390, 378])
ySpeed02 = np.asarray([346, 349, 352, 355, 358, 360, 362, 364, 365, 367, 368, 369])

xSpeed03 = np.asarray([504, 488, 472, 456, 440, 425, 410, 395, 380, 366, 352, 337])
ySpeed03 = np.asarray([369, 370, 372, 373, 374, 375, 376, 377, 378, 378, 379, 379])

xSpeed04 = np.asarray([315, 337, 359, 379, 400, 420, 440, 460, 480, 499, 519, 538])
ySpeed04 = np.asarray([427, 430, 431, 432, 433, 434, 435, 435, 436, 436, 437, 437])

xSpeed05 = np.asarray([257, 279, 299, 318, 337, 355, 373, 390, 408, 425, 442, 459])
ySpeed05 = np.asarray([395, 393, 392, 390, 389, 387, 386, 384, 383, 382, 380, 379])

xSpeed06 = np.asarray([575, 563, 550, 537, 523, 510, 496, 482, 468, 453, 439, 425])
ySpeed06 = np.asarray([328, 336, 344, 351, 357, 362, 367, 371, 375, 379, 382, 385])

xSpeed07 = np.asarray([577, 558, 538, 517, 496, 476, 455, 435, 415, 395, 375, 355])
ySpeed07 = np.asarray([359, 366, 373, 378, 383, 387, 391, 394, 397, 400, 403, 406])

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
