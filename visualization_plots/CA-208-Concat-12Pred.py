import numpy as np
import matplotlib.pyplot as plt
#[67 68 70 71 73 74 75 76 78 79 80 81]
#[453 454 455 455 455 456 456 456 457 457 457 458]

xSpeed01 = np.asarray([487, 472, 457, 442, 428, 413, 399, 385, 371, 356, 343, 329])
ySpeed01 = np.asarray([403, 400, 397, 395, 392, 390, 387, 385, 383, 381, 379, 377])

xSpeed02 = np.asarray([510, 495, 481, 468, 454, 441, 427, 414, 401, 387, 374, 361])
ySpeed02 = np.asarray([343, 342, 341, 339, 338, 336, 334, 333, 331, 329, 328, 326])

xSpeed03 = np.asarray([503, 487, 472, 456, 441, 426, 411, 396, 381, 367, 352, 338])
ySpeed03 = np.asarray([366, 365, 364, 363, 361, 360, 359, 357, 356, 355, 353, 352])

xSpeed04 = np.asarray([315, 337, 359, 381, 404, 427, 450, 473, 496, 520, 543, 567])
ySpeed04 = np.asarray([427, 430, 433, 436, 439, 441, 443, 445, 446, 448, 449, 450])

xSpeed05 = np.asarray([257, 280, 302, 324, 347, 369, 392, 414, 437, 460, 482, 506])
ySpeed05 = np.asarray([394, 395, 395, 395, 396, 396, 396, 396, 397, 397, 397, 397])

xSpeed06 = np.asarray([573, 559, 544, 529, 514, 499, 483, 468, 453, 439, 424, 409])
ySpeed06 = np.asarray([324, 328, 332, 335, 338, 342, 345, 348, 351, 354, 357, 360])

xSpeed07 = np.asarray([575, 555, 536, 515, 495, 475, 455, 435, 416, 397, 377, 358])
ySpeed07 = np.asarray([356, 360, 363, 366, 369, 372, 375, 378, 381, 385, 388, 392])

#xSpeed08 = np.asarray()
#ySpeed08 = np.asarray()

plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='orange')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='orange')
plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='orange')
plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='red')
plt.plot(xSpeed05, ySpeed05, zorder=1, linewidth=3, linestyle='--', color='yellow')
#plt.plot(xSpeed06, ySpeed06, zorder=1, linewidth=3, linestyle='--', color='blue')
#plt.plot(xSpeed07, ySpeed07, zorder=1, linewidth=3, linestyle='--', color='blue')

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='orange')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='orange')
plt.plot(xSpeed03[-1], ySpeed03[-1], zorder=1, marker='<', color='orange')
plt.plot(xSpeed04[-1], ySpeed04[-1], zorder=1, marker='>', color='red')
plt.plot(xSpeed05[-1], ySpeed05[-1], zorder=1, marker='>', color='yellow')
#plt.plot(xSpeed06[-1], ySpeed06[-1], zorder=1, marker='>', color='purple')

#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-208.png")
plt.imshow(img, zorder=0)
plt.show()
