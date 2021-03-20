import numpy as np
import matplotlib.pyplot as plt

xSpeed01 = np.asarray([409, 396, 386, 378, 371, 364, 359, 353, 348, 344, 340, 335])
ySpeed01 = np.asarray([385, 383, 381, 378, 376, 372, 369, 366, 363, 359, 356, 353])
xSpeed02 = np.asarray([400, 387, 377, 369, 362, 356, 351, 346, 342, 337, 333, 329])
ySpeed02 = np.asarray([409, 406, 402, 398, 393, 389, 385, 380, 376, 371, 367, 363])
xSpeed03 = np.asarray([322, 341, 355, 366, 375, 382, 388, 393, 398, 402, 405, 408])
ySpeed03 = np.asarray([331, 335, 337, 339, 341, 343, 345, 346, 348, 349, 350, 351])
xSpeed04 = np.asarray([215, 233, 246, 257, 266, 274, 280, 286, 290, 295, 299, 302])
ySpeed04 = np.asarray([419, 428, 436, 443, 450, 456, 462, 468, 474, 479, 484, 488])
xSpeed05 = np.asarray([455, 443, 434, 426, 419, 414, 408, 404, 399, 395, 391, 387])
ySpeed05 = np.asarray([316, 313, 310, 307, 304, 301, 298, 296, 293, 290, 288, 286])
xSpeed06 = np.asarray([483, 469, 459, 450, 443, 437, 431, 426, 422, 417, 413, 409])
ySpeed06 = np.asarray([349, 347, 344, 342, 339, 337, 335, 332, 330, 328, 326, 324])

xSpeedGT01 = np.asarray([404, 383, 362, 340, 320, 300, 280, 259, 236, 213, 190, 168])
ySpeedGT01 = np.asarray([384, 383, 382, 379, 376, 373, 369, 366, 365, 364, 363, 365])

xSpeedGT02 = np.asarray([395, 372, 351, 330, 309, 288, 267, 245, 224, 202, 180, 158])
ySpeedGT02 = np.asarray([410, 409, 408, 407, 407, 406, 406, 403, 399, 395, 397, 399])

xSpeedGT03 = np.asarray([321, 342, 363, 388, 413, 438, 463, 488, 514, 539, 564, 588])
ySpeedGT03 = np.asarray([334, 341, 348, 350, 352, 354, 357, 360, 362, 365, 368, 371])

xSpeedGT04 = np.asarray([217, 241, 265, 294, 323, 350, 377, 404, 430, 458, 485, 507])
ySpeedGT04 = np.asarray([417, 426, 434, 438, 442, 439, 433, 427, 422, 416, 411, 407])

xSpeedGT05 = np.asarray([452, 434, 414, 393, 372, 347, 323, 298, 273, 249, 224, 201])
ySpeedGT05 = np.asarray([315, 309, 304, 298, 292, 292, 291, 291, 291, 291, 291, 290])

xSpeedGT06 = np.asarray([478, 457, 436, 416, 393, 369, 346, 323, 301, 279, 257, 235])
ySpeedGT06 = np.asarray([348, 340, 330, 320, 318, 318, 318, 319, 319, 319, 319, 321])

plt.plot(xSpeed01,ySpeed01,zorder=1, color='red', linewidth=3, linestyle='--', label='0.1')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed05,ySpeed05,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed06,ySpeed06,zorder=1, color='red', linewidth=3, linestyle='--')

plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', linewidth=3, linestyle='--', label='GT')
plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT05,ySpeedGT05,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT06,ySpeedGT06,zorder=1, color='blue', linewidth=3, linestyle='--')


plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, color='red', marker='>', linewidth=3, linestyle='--')
plt.plot(xSpeed05[-1],ySpeed05[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeed06[-1],ySpeed06[-1],zorder=1, color='red', marker='<', linewidth=3, linestyle='--')

plt.plot(xSpeedGT01[-1],ySpeedGT01[-1],zorder=1, color='blue', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeedGT02[-1],ySpeedGT02[-1],zorder=1, color='blue', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeedGT03[-1],ySpeedGT03[-1],zorder=1, color='blue', marker='>', linewidth=3, linestyle='--')
plt.plot(xSpeedGT05[-1],ySpeedGT05[-1],zorder=1, color='blue', marker='<', linewidth=3, linestyle='--')
plt.plot(xSpeedGT06[-1],ySpeedGT06[-1],zorder=1, color='blue', marker='<', linewidth=3, linestyle='--')

#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara150.png")
plt.imshow(img, zorder=0)
plt.show()
