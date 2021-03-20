import numpy as np
import matplotlib.pyplot as plt



xSpeed01 = np.asarray([488, 473, 458, 444, 429, 417, 405, 394, 382, 372, 362, 353])
ySpeed01 = np.asarray([404, 402, 399, 397, 395, 392, 389, 385, 382, 379, 376, 374])

xSpeed02 = np.asarray([512, 500, 489, 477, 465, 454, 444, 434, 424, 414, 404, 393])
ySpeed02 = np.asarray([343, 340, 338, 336, 334, 332, 330, 328, 327, 325, 321, 314])

xSpeed03 = np.asarray([504, 493, 482, 470, 459, 448, 437, 425, 414, 404, 393, 382])
ySpeed03 = np.asarray([368, 368, 368, 368, 366, 364, 362, 360, 358, 354, 350, 346])

xSpeed04 = np.asarray([314, 335, 358, 380, 403, 425, 447, 469, 490, 512, 534, 556])
ySpeed04 = np.asarray([427, 430, 430, 430, 430, 430, 430, 429, 428, 427, 421, 415])

xSpeed05 = np.asarray([256, 277, 299, 320, 336, 349, 363, 376, 391, 406, 421, 438])
ySpeed05 = np.asarray([397, 404, 411, 418, 424, 429, 434, 439, 443, 446, 449, 452])

xSpeed06 = np.asarray([572, 558, 544, 530, 516, 502, 489, 476, 463, 449, 434, 420])
ySpeed06 = np.asarray([325, 328, 332, 337, 343, 349, 357, 366, 375, 375, 374, 373])

xSpeed07 = np.asarray([573, 559, 545, 532, 518, 506, 494, 481, 466, 451, 435, 418])
ySpeed07 = np.asarray([358, 366, 375, 383, 392, 396, 400, 404, 407, 410, 412, 412])

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
