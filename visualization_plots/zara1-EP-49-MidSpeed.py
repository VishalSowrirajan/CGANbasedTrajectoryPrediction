import numpy as np
import matplotlib.pyplot as plt

xSpeed01 = np.asarray([101, 105, 110, 114, 117, 120, 123, 125, 127, 129, 131, 133])
ySpeed01 = np.asarray([282, 280, 279, 278, 277, 276, 275, 274, 274, 273, 273, 273])

xSpeed02 = np.asarray([340, 317, 294, 273, 252, 233, 215, 197, 181, 165, 149, 134])
ySpeed02 = np.asarray([368, 374, 378, 382, 385, 388, 391, 393, 395, 396, 398, 399])

xSpeed03 = np.asarray([352, 330, 308, 288, 268, 250, 232, 215, 198, 182, 167, 152])
ySpeed03 = np.asarray([453, 453, 453, 453, 452, 451, 449, 447, 446, 443, 441, 438])

xSpeed04 = np.asarray([357, 338, 319, 301, 283, 265, 248, 232, 216, 201, 186, 171])
ySpeed04 = np.asarray([413, 414, 414, 414, 413, 413, 412, 411, 409, 408, 406, 403])

xSpeed05 = np.asarray([236, 258, 280, 300, 319, 337, 354, 370, 386, 401, 415, 429])
ySpeed05 = np.asarray([379, 375, 371, 369, 366, 364, 362, 359, 357, 355, 354, 352])

xSpeedGT01 = np.asarray([107, 119, 131, 143, 154, 165, 176, 184, 191, 198, 205, 212])
ySpeedGT01 = np.asarray([283, 282, 282, 281, 280, 279, 278, 277, 277, 276, 276, 275])

xSpeedGT02 = np.asarray([343, 322, 300, 279, 258, 237, 214, 191, 168, 145, 124, 103])
ySpeedGT02 = np.asarray([369, 377, 383, 384, 386, 387, 388, 389, 390, 391, 393, 396])

xSpeedGT03 = np.asarray([350, 327, 304, 282, 259, 236, 213, 190, 166, 143, 118,  93])
ySpeedGT03 = np.asarray([451, 450, 450, 449, 449, 449, 449, 449, 449, 450, 451, 453])

xSpeedGT04 = np.asarray([353, 330, 307, 285, 262, 240, 217, 195, 173, 151, 129, 106])
ySpeedGT04 = np.asarray([412, 411, 411, 412, 413, 415, 416, 416, 416, 417, 417, 418])

xSpeedGT05 = np.asarray([238, 264, 290, 315, 341, 366, 391, 415, 440, 464, 490, 515])
ySpeedGT05 = np.asarray([377, 368, 365, 362, 362, 362, 362, 368, 373, 379, 389, 398])

#plt.plot(xSpeed01,ySpeed01,zorder=1, color='purple', label='0.5', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='red', linewidth=3, linestyle='--')
plt.plot(xSpeed05,ySpeed05,zorder=1, color='red', linewidth=3, linestyle='--')

#plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT04,ySpeedGT04,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT05,ySpeedGT05,zorder=1, color='blue', linewidth=3, linestyle='--')


#plt.plot(xSpeed03,ySpeed03,zorder=1, color='purple', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeed04,ySpeed04,zorder=1, color='purple', linewidth=3, linestyle='--')

#plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT04,ySpeedGT04,zorder=1, color='blue', linewidth=3, linestyle='--')


#plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='<', color='purple', linewidth=2)
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='<', color='red', linewidth=2)
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, marker='<', color='red', linewidth=2)
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, marker='<', color='red', linewidth=2)
plt.plot(xSpeed05[-1],ySpeed05[-1],zorder=1, marker='>', color='red', linewidth=2)
#plt.plot(xSpeedGT01[-1],ySpeedGT01[-1],zorder=1, marker='<', color='blue', linewidth=2)
plt.plot(xSpeedGT02[-1],ySpeedGT02[-1],zorder=1, marker='<', color='blue', linewidth=2)
plt.plot(xSpeedGT03[-1],ySpeedGT03[-1],zorder=1, marker='<', color='blue', linewidth=2)
plt.plot(xSpeedGT04[-1],ySpeedGT04[-1],zorder=1, marker='<', color='blue', linewidth=2)
plt.plot(xSpeedGT05[-1],ySpeedGT05[-1],zorder=1, marker='>', color='blue', linewidth=2)

#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-48.png")
plt.imshow(img, zorder=0)
plt.show()
