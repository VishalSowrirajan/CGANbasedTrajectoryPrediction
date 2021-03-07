import numpy as np
import matplotlib.pyplot as plt

# PLOTTING HOTEL DATASET - Plot 1 in figure 3


xSpeed01 = np.asarray([507, 486, 466, 444, 422, 399, 374, 349, 323, 296, 268, 240])
ySpeed01 = np.asarray([317, 320, 324, 329, 334, 339, 345, 351, 357, 363, 369, 374])

xSpeed02 = np.asarray([515, 495, 476, 455, 433, 410, 386, 361, 335, 308, 281, 253])
ySpeed02 = np.asarray([288, 292, 297, 303, 309, 316, 323, 330, 337, 343, 349, 354])

xSpeedGT = np.asarray([508, 492, 473, 455, 436, 419, 402, 384, 366, 346, 322, 299])
ySpeedGT = np.asarray([323, 328, 332, 338, 343, 346, 352, 357, 359, 362, 369, 371])

xSpeedGT02 = np.asarray([516, 500, 483, 464, 444, 425, 406, 388, 368, 349, 331, 312])
ySpeedGT02 = np.asarray([284, 284, 287, 286, 292, 295, 298, 304, 307, 314, 318, 324])

xSpeed03 = np.asarray([508, 490, 473, 455, 437, 419, 400, 380, 361, 340, 320, 299])
ySpeed03 = np.asarray([317, 320, 324, 328, 332, 336, 338, 341, 342, 343, 343, 343])

xSpeed04 = np.asarray([516, 499, 482, 465, 448, 430, 411, 392, 373, 353, 333, 312])
ySpeed04 = np.asarray([287, 291, 295, 300, 305, 309, 313, 315, 317, 318, 319, 319])

xSpeed05 = np.asarray([507, 490, 475, 461, 447, 435, 422, 411, 399, 388, 378, 368])
ySpeed05 = np.asarray([320, 321, 323, 323, 324, 323, 323, 323, 323, 323, 324, 326])

xSpeed06 = np.asarray([516, 500, 485, 471, 459, 446, 434, 423, 412, 402, 391, 382])
ySpeed06 = np.asarray([290, 292, 294, 295, 296, 296, 296, 296, 297, 297, 299, 301])


plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red', label='0.9')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='red')

#plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='cyan', label='0.5')
#plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='cyan')

plt.plot(xSpeed05, ySpeed05, zorder=1, linewidth=3, linestyle='--', color='purple', label='0.2')
plt.plot(xSpeed06, ySpeed06, zorder=1, linewidth=3, linestyle='--', color='purple')

plt.plot(xSpeedGT, ySpeedGT, zorder=1, linewidth=3, color='blue', label='GT', linestyle='--')
plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, linewidth=3, color='blue', linestyle='--')

plt.plot(xSpeed01[-1], ySpeed01[-1], zorder=1, marker='<', color='red')
plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='red')

plt.plot(xSpeedGT[-1], ySpeedGT[-1], zorder=1, marker='<', color='blue')
plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='<', color='blue')

plt.plot(xSpeed05[-1], ySpeed05[-1], zorder=1, marker='<', color='purple')
plt.plot(xSpeed06[-1], ySpeed06[-1], zorder=1, marker='<', color='purple')

plt.legend(loc='center right')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel22.png")
plt.imshow(img, zorder=0)
plt.show()
