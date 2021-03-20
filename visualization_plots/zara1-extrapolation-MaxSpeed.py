import numpy as np
import matplotlib.pyplot as plt
#[ 126  111   92   71   49   25    0  -27  -54  -82 -109 -137]
#[373 392 409 423 436 446 453 459 464 469 474 479]
#[196 211 231 256 284 315 347 379 411 444 476 509]
#[299 305 310 314 318 322 324 327 328 329 329 328]

xSpeed01 = np.asarray([254, 274, 299, 327, 357, 386, 416, 445, 475, 504, 534, 563])
ySpeed01 = np.asarray([277, 284, 291, 299, 308, 318, 328, 338, 349, 360, 372, 383])

xSpeed02 = np.asarray([251, 273, 301, 331, 361, 391, 421, 450, 480, 509, 539, 569])
ySpeed02 = np.asarray([302, 309, 316, 324, 332, 342, 352, 363, 374, 385, 397, 408])

xSpeed03 = np.asarray([130, 131, 134, 136, 138, 139, 139, 139, 138, 137, 134, 131])
ySpeed03 = np.asarray([371, 392, 417, 443, 469, 494, 518, 541, 565, 588, 611, 635])

xSpeed04 = np.asarray([203, 231, 262, 295, 327, 359, 391, 423, 456, 488, 521, 553])
ySpeed04 = np.asarray([297, 300, 303, 306, 308, 309, 309, 309, 308, 306, 303, 301])

xSpeedGT01 = np.asarray([253, 268, 282, 296, 309, 322, 338, 355, 371, 388, 406, 423])
ySpeedGT01 = np.asarray([273, 275, 276, 278, 279, 280, 287, 293, 300, 307, 315, 323])

xSpeedGT02 = np.asarray([248, 262, 277, 292, 306, 321, 334, 348, 362, 376, 392, 408])
ySpeedGT02 = np.asarray([298, 298, 298, 299, 300, 302, 307, 311, 316, 321, 327, 333])

xSpeedGT03 = np.asarray([124, 113, 102,  91,  81,  70,  61,  54,  46,  39,  33,  27])
ySpeedGT03 = np.asarray([372, 390, 407, 425, 443, 461, 479, 495, 511, 527, 539, 545])

xSpeedGT04 = np.asarray([197, 212, 228, 244, 259, 272, 285, 297, 308, 320, 333, 346])
ySpeedGT04 = np.asarray([296, 296, 296, 296, 295, 294, 292, 292, 294, 296, 298, 301])


plt.plot(xSpeed01,ySpeed01,zorder=1, color='purple', label='0.9', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='purple', linewidth=3, linestyle='--')
#plt.plot(xSpeed03,ySpeed03,zorder=1, color='purple', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeed04,ySpeed04,zorder=1, color='purple', linewidth=3, linestyle='--')

plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT04,ySpeedGT04,zorder=1, color='blue', linewidth=3, linestyle='--')


plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='>', color='purple', linewidth=2)
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='>', color='purple', linewidth=2)
plt.plot(xSpeedGT01[-1],ySpeedGT01[-1],zorder=1, marker='>', color='blue', linewidth=2)
plt.plot(xSpeedGT02[-1],ySpeedGT02[-1],zorder=1, marker='>', color='blue', linewidth=2)

plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-29.png")
plt.imshow(img, zorder=0)
plt.show()
