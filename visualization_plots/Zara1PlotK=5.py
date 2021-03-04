import numpy as np
import matplotlib.pyplot as plt

### K=5 ZARA1 output at 40th second ###

xSpeed01 = np.asarray([524, 501, 479, 455, 433, 410, 388, 367, 345, 323, 302, 280])
ySpeed01 = np.asarray([322, 326, 329, 331, 331, 331, 331, 331, 330, 329, 328, 327])

xSpeedGT = np.asarray([525, 497, 468, 440, 412, 384, 356, 329, 301, 272, 243, 215])
ySpeedGT = np.asarray([327, 332, 338, 342, 345, 348, 352, 356, 360, 363, 365, 367])

xSpeed02 = np.asarray([526, 505, 484, 463, 442, 421, 401, 381, 361, 340, 320, 299])
ySpeed02 = np.asarray([325, 333, 339, 345, 349, 353, 357, 361, 364, 367, 370, 372])

xSpeed03 = np.asarray([528, 509, 491, 472, 452, 433, 413, 393, 373, 353, 332, 311])
ySpeed03 = np.asarray([325, 332, 338, 341, 343, 345, 346, 347, 348, 348, 348, 349])

xSpeed04 = np.asarray([528, 508, 487, 467, 446, 426, 406, 386, 366, 346, 325, 305])
ySpeed04 = np.asarray([325, 331, 335, 339, 342, 345, 347, 349, 351, 352, 354, 355])

xSpeed05 = np.asarray([527, 508, 489, 469, 450, 431, 413, 394, 375, 357, 338, 319])
ySpeed05 = np.asarray([326, 335, 343, 350, 357, 363, 369, 375, 381, 386, 391, 395])

xSpeed06 = np.asarray([526, 504, 482, 459, 437, 415, 393, 351, 330, 309, 287])
ySpeed06 = np.asarray([326, 335, 341, 347, 351, 357, 360, 362, 365, 367, 368])


plt.plot(xSpeedGT,ySpeedGT,zorder=1, color='blue', linestyle='--', linewidth=2, label='GT')
plt.plot(xSpeed01,ySpeed01,zorder=1, color='yellow', linestyle='--', linewidth=2, label='K=1')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='green', linestyle='--', linewidth=2, label='K=2')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='red', linestyle='--', linewidth=2, label='K=3')
plt.plot(xSpeed05,ySpeed05,zorder=1, color='brown', linestyle='--', linewidth=2, label='K=4')
plt.plot(xSpeed06,ySpeed06,zorder=1, color='orange', linestyle='--', linewidth=2, label='K=5')

plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, color='yellow', marker='<')
plt.plot(xSpeedGT[-1],ySpeedGT[-1],zorder=1, color='blue', marker='<')
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, color='green', marker='<')
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, color='red', marker='<')
plt.plot(xSpeed05[-1],ySpeed05[-1],zorder=1, color='brown', marker='<')
plt.plot(xSpeed06[-1],ySpeed06[-1],zorder=1, color='orange', marker='<')


plt.xticks([])
plt.legend()
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara40-1.png")
plt.imshow(img, zorder=0)
plt.show()
