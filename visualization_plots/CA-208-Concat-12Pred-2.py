import numpy as np
import matplotlib.pyplot as plt

#[64 65 67 68 70 71 72 73 74 74 75 76]
#[452 452 452 453 453 454 455 455 456 457 458 459]

xSpeed01 = np.asarray([487, 472, 457, 443, 429, 415, 400, 386, 372, 357, 343, 329])
ySpeed01 = np.asarray([402, 398, 394, 390, 387, 384, 381, 378, 375, 373, 371, 369])

xSpeed02 = np.asarray([509, 495, 481, 467, 454, 440, 427, 413, 400, 386, 373, 359])
ySpeed02 = np.asarray([342, 339, 337, 334, 331, 328, 325, 323, 321, 319, 317, 316])

xSpeed03 = np.asarray([503, 487, 471, 456, 441, 425, 410, 395, 380, 365, 350, 335])
ySpeed03 = np.asarray([365, 363, 360, 358, 355, 353, 351, 348, 347, 345, 343, 341])

xSpeed04 = np.asarray([315, 337, 359, 381, 404, 427, 450, 473, 496, 519, 542, 565])
ySpeed04 = np.asarray([427, 429, 433, 438, 443, 449, 455, 461, 467, 474, 480, 486])

xSpeed05 = np.asarray([258, 281, 303, 326, 349, 372, 395, 418, 441, 463, 486, 509])
ySpeed05 = np.asarray([394, 394, 394, 396, 399, 402, 405, 409, 414, 419, 423, 428])

xSpeed06 = np.asarray([574, 560, 545, 529, 513, 497, 482, 467, 451, 436, 421, 406])
ySpeed06 = np.asarray([324, 327, 329, 330, 331, 331, 332, 332, 332, 332, 332, 332])

xSpeed07 = np.asarray([576, 557, 537, 517, 497, 477, 457, 438, 418, 399, 379, 360])
ySpeed07 = np.asarray([356, 359, 360, 361, 361, 361, 361, 360, 360, 359, 359, 358])

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
