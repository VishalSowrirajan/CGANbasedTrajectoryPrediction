import numpy as np
import matplotlib.pyplot as plt

xSpeed01 = np.asarray([499, 472, 442, 410, 379, 347, 315, 283, 251, 219, 187, 155])
ySpeed01 = np.asarray([332, 332, 332, 330, 329, 327, 326, 324, 322, 321, 319, 317])
xSpeed02 = np.asarray([492, 464, 434, 402, 370, 338, 306, 274, 242, 210, 178, 146])
ySpeed02 = np.asarray([434, 435, 436, 435, 435, 434, 434, 433, 432, 432, 431, 431])
xSpeed03 = np.asarray([518, 490, 460, 428, 396, 365, 333, 300, 268, 236, 204, 172])
ySpeed03 = np.asarray([397, 399, 401, 401, 401, 402, 402, 402, 402, 403, 403, 404])
xSpeed04 = np.asarray([516, 489, 459, 428, 396, 364, 332, 300, 268, 236, 204, 173])
ySpeed04 = np.asarray([361, 361, 361, 359, 358, 356, 355, 353, 351, 350, 348, 346])

xSpeedGT01 = np.asarray([500, 480, 463, 445, 428, 408, 387, 365, 344, 324, 305, 286])
ySpeedGT01 = np.asarray([330, 329, 328, 326, 325, 323, 320, 317, 315, 312, 309, 307])
xSpeedGT02 = np.asarray([493, 473, 451, 429, 408, 386, 365, 346, 327, 308, 289, 270])
ySpeedGT02 = np.asarray([432, 432, 432, 432, 432, 432, 431, 430, 429, 428, 427, 425])
xSpeedGT03 = np.asarray([519, 499, 478, 458, 438, 418, 398, 378, 358, 338, 318, 298])
ySpeedGT03 = np.asarray([395, 396, 397, 396, 394, 393, 391, 389, 386, 384, 382, 380])
xSpeedGT04 = np.asarray([519, 500, 481, 461, 440, 418, 396, 374, 352, 333, 317, 300])
ySpeedGT04 = np.asarray([359, 358, 357, 357, 355, 353, 351, 348, 346, 344, 340, 337])


plt.plot(xSpeed01,ySpeed01,zorder=1, color='red', linewidth=3, linestyle='--', label='0.9')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='red', linewidth=3, linestyle='--')

plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', linewidth=3, linestyle='--', label='GT')
plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT04,ySpeedGT04,zorder=1, color='blue', linewidth=3, linestyle='--')


plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')


plt.plot(xSpeedGT01[-1],ySpeedGT01[-1],zorder=1, color='blue', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeedGT02[-1],ySpeedGT02[-1],zorder=1, color='blue', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeedGT03[-1],ySpeedGT03[-1],zorder=1, color='blue', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeedGT04[-1],ySpeedGT04[-1],zorder=1, color='blue', marker='<', linewidth=3, linestyle='--')


#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-297.png")
plt.imshow(img, zorder=0)
plt.show()
