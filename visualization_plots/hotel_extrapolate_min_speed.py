import numpy as np
import matplotlib.pyplot as plt


xSpeed01 = np.asarray([96,  97,  99, 102, 104, 106, 108, 110])
ySpeed01 = np.asarray([378, 385, 391, 395, 399, 402, 405, 407])

xSpeed02 = np.asarray([44, 44, 46, 47, 48, 49, 51, 52])
ySpeed02 = np.asarray([371, 378, 383, 387, 391, 394, 396, 398])

xSpeed03 = np.asarray([157, 158, 159, 160, 162, 163, 165, 166])
ySpeed03 = np.asarray([283, 289, 294, 298, 302, 305, 307, 309])

xSpeed04 = np.asarray([389, 368, 350, 332, 316, 299, 284, 269])
ySpeed04 = np.asarray([379, 389, 399, 408, 418, 427, 436, 446])

xSpeed05 = np.asarray([379, 400, 419, 436, 453, 468, 482, 495])
ySpeed05 = np.asarray([350, 345, 340, 336, 332, 328, 323, 319])

xSpeedGT01 = np.asarray([440, 440, 440, 440, 440, 440, 439, 434, 427, 427, 418, 420])
ySpeedGT01 = np.asarray([215, 215, 215, 215, 215, 215, 215, 212, 206, 206, 188, 185])

xSpeedGT02 = np.asarray([516, 490, 467, 444, 419, 395, 374, 352, 330, 306, 280, 251])
ySpeedGT02 = np.asarray([359, 357, 359, 362, 363, 369, 376, 380, 384, 390, 391, 390])

xSpeedGT03 = np.asarray([544, 523, 495, 473, 452, 420, 399, 379, 357, 334, 310, 284])
ySpeedGT03 = np.asarray([323, 326, 323, 329, 337, 336, 340, 352, 354, 356, 365, 359])


plt.plot(xSpeed01, ySpeed01, zorder=1, linewidth=3, linestyle='--', color='red', label='0.6')
plt.plot(xSpeed02, ySpeed02, zorder=1, linewidth=3, linestyle='--', color='red')
plt.plot(xSpeed03, ySpeed03, zorder=1, linewidth=3, linestyle='--', color='red')
plt.plot(xSpeed04, ySpeed04, zorder=1, linewidth=3, linestyle='--', color='red')
plt.plot(xSpeed05, ySpeed05, zorder=1, linewidth=3, linestyle='--', color='red')

#plt.plot(xSpeedGT01, ySpeedGT01, zorder=1, linewidth=3, linestyle='--', color='blue', label='GT')
#plt.plot(xSpeedGT02, ySpeedGT02, zorder=1, linewidth=3, linestyle='--', color='blue')
#plt.plot(xSpeedGT03, ySpeedGT03, zorder=1, linewidth=3, linestyle='--', color='blue')

#plt.plot(xSpeed02[-1], ySpeed02[-1], zorder=1, marker='<', color='red')
#plt.plot(xSpeed03[-1], ySpeed03[-1], zorder=1, marker='<', color='red')

#plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='<', color='blue')
#plt.plot(xSpeedGT03[-1], ySpeedGT03[-1], zorder=1, marker='<', color='blue')

#plt.plot(xSpeedGT[-1], ySpeedGT[-1], zorder=1, marker='<', color='blue')
#plt.plot(xSpeedGT02[-1], ySpeedGT02[-1], zorder=1, marker='<', color='blue')

#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../hotel404.png")
plt.imshow(img, zorder=0)
plt.show()
