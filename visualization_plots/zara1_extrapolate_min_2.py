import numpy as np
import matplotlib.pyplot as plt

xSpeed01 = np.asarray([442, 429, 421, 416, 413, 411, 409, 407, 406, 404, 403, 401])
ySpeed01 = np.asarray([372, 369, 366, 362, 360, 357, 355, 354, 352, 351, 349, 348])

xSpeed02 = np.asarray([445, 432, 423, 418, 415, 413, 412, 410, 409, 407, 406, 404])
ySpeed02 = np.asarray([337, 334, 330, 327, 324, 321, 319, 317, 316, 314, 313, 311])


xSpeed03 = np.asarray([435, 410, 385, 360, 334, 309, 284, 259, 234, 209, 185, 160])
ySpeed03 = np.asarray([374, 373, 373, 372, 372, 374, 378, 383, 387, 393, 399, 405])

xSpeed04 = np.asarray([438, 414, 390, 366, 343, 319, 296, 274, 250, 224, 198, 172])
ySpeed04 = np.asarray([339, 339, 339, 339, 339, 339, 338, 336, 338, 344, 351, 358])


plt.plot(xSpeed03,ySpeed03,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeed01,ySpeed01,zorder=1, color='red', label='0.0', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='red', linewidth=3, linestyle='--')

plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='>', color='red', linewidth=2)
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='>', color='red', linewidth=2)
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, marker='>', color='blue', linewidth=2)
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, marker='>', color='blue', linewidth=2)

plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-105.png")
plt.imshow(img, zorder=0)
plt.show()
