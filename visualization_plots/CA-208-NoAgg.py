import numpy as np
import matplotlib.pyplot as plt


#[63 63 64 64 65 65 65 65 66 66 66 66]
#[452 451 451 451 451 451 451 451 451 451 452 452]

xSpeed01 = np.asarray([487, 473, 459, 446, 433, 421, 409, 397, 386, 375, 364, 353])
ySpeed01 = np.asarray([404, 401, 399, 397, 395, 392, 390, 388, 386, 384, 382, 380])

xSpeed02 = np.asarray([511, 499, 487, 476, 465, 454, 444, 434, 425, 415, 406, 396])
ySpeed02 = np.asarray([343, 343, 343, 343, 342, 341, 340, 339, 338, 337, 336, 334])

xSpeed03 = np.asarray([503, 488, 473, 459, 446, 433, 420, 408, 396, 384, 372, 360])
ySpeed03 = np.asarray([367, 366, 365, 363, 362, 360, 358, 357, 355, 353, 352, 350])

xSpeed04 = np.asarray([315, 337, 359, 380, 401, 422, 443, 464, 484, 505, 526, 547])
ySpeed04 = np.asarray([425, 427, 429, 432, 435, 437, 439, 441, 443, 445, 447, 448])

xSpeed05 = np.asarray([257, 279, 301, 322, 343, 364, 385, 405, 426, 447, 467, 488])
ySpeed05 = np.asarray([392, 391, 391, 392, 392, 393, 394, 394, 395, 396, 396, 397])

xSpeed06 = np.asarray([572, 558, 545, 532, 519, 506, 493, 479, 466, 452, 439, 425])
ySpeed06 = np.asarray([324, 329, 332, 335, 337, 339, 341, 342, 343, 343, 343, 344])

xSpeed07 = np.asarray([572, 552, 532, 511, 491, 470, 449, 429, 409, 389, 369, 349])
ySpeed07 = np.asarray([356, 360, 363, 365, 367, 369, 370, 371, 371, 372, 372, 372])

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
