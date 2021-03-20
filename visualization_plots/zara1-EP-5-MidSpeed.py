import numpy as np
import matplotlib.pyplot as plt

xSpeed01 = np.asarray([331, 307, 284, 263, 242, 222, 203, 185, 168, 151, 135, 120])
ySpeed01 = np.asarray([445, 449, 453, 456, 459, 462, 465, 467, 469, 471, 473, 474])

xSpeed02 = np.asarray([325, 301, 279, 257, 236, 216, 198, 180, 162, 146, 130, 114])
ySpeed02 = np.asarray([421, 425, 429, 433, 436, 439, 441, 444, 446, 448, 449, 451])

xSpeed03 = np.asarray([312, 295, 277, 260, 243, 227, 211, 196, 180, 166, 151, 137])
ySpeed03 = np.asarray([364, 363, 362, 362, 361, 360, 359, 358, 356, 354, 352, 350])

xSpeed04 = np.asarray([300, 282, 264, 246, 229, 212, 196, 180, 164, 149, 134, 120])
ySpeed04 = np.asarray([342, 341, 340, 339, 339, 338, 337, 335, 334, 332, 330, 328])

xSpeed05 = np.asarray([180, 157, 135, 115,  95,  76,  58,  40,  24,   8,  -7, -22])
ySpeed05 = np.asarray([436, 440, 444, 448, 451, 454, 457, 460, 463, 465, 467, 469])

xSpeed06 = np.asarray([616, 626, 635, 643, 651, 658, 664, 670, 676, 681, 686, 691])
ySpeed06 = np.asarray([346, 332, 319, 308, 298, 288, 278, 269, 260, 252, 243, 235])

xSpeed07 = np.asarray([296, 297, 298, 298, 298, 297, 296, 295, 294, 292, 290, 289])
ySpeed07 = np.asarray([264, 261, 259, 258, 256, 255, 253, 252, 251, 250, 249, 248])

xSpeed08 = np.asarray([470, 448, 426, 406, 386, 367, 349, 332, 315, 299, 284, 269])
ySpeed08 = np.asarray([452, 457, 462, 466, 470, 474, 478, 481, 484, 487, 489, 491])

xSpeedGT01 = np.asarray([331, 306, 280, 254, 228, 204, 180, 157, 133, 112,  90,  68])
ySpeedGT01 = np.asarray([442, 444, 445, 446, 447, 450, 455, 459, 463, 465, 468, 471])

xSpeedGT02 = np.asarray([326, 301, 277, 254, 230, 206, 181, 156, 132, 110,  87,  65])
ySpeedGT02 = np.asarray([418, 420, 421, 421, 421, 422, 425, 428, 430, 432, 434, 437])

xSpeedGT03 = np.asarray([311, 292, 273, 253, 235, 217, 199, 181, 163, 145, 126, 108])
ySpeedGT03 = np.asarray([364, 364, 364, 364, 363, 362, 362, 361, 359, 356, 352, 349])

xSpeedGT04 = np.asarray([300, 281, 263, 245, 227, 209, 192, 175, 158, 142, 128, 114])
ySpeedGT04 = np.asarray([340, 339, 337, 335, 333, 332, 329, 327, 325, 322, 320, 318])

xSpeedGT05 = np.asarray([186, 168, 151, 133, 116,  99,  83,  67,  51,  35,  18,   2])
ySpeedGT05 = np.asarray([434, 436, 438, 440, 442, 442, 442, 442, 442, 442, 442, 442])

xSpeedGT06 = np.asarray([621, 633, 638, 643, 648, 653, 653, 653, 652, 651, 651, 650])
ySpeedGT06 = np.asarray([346, 329, 309, 289, 271, 252, 234, 217, 199, 182, 165, 149])

xSpeedGT07 = np.asarray([298, 298, 297, 295, 284, 273, 262, 251, 242, 232, 223, 214])
ySpeedGT07 = np.asarray([265, 263, 260, 259, 259, 260, 261, 261, 260, 258, 257, 257])

xSpeedGT08 = np.asarray([469, 443, 418, 397, 376, 354, 332, 309, 287, 264, 242, 220])
ySpeedGT08 = np.asarray([449, 452, 454, 452, 450, 449, 450, 451, 452, 452, 450, 448])

plt.plot(xSpeed01,ySpeed01,zorder=1, color='purple', label='0.5', linewidth=3, linestyle='--')
plt.plot(xSpeed02,ySpeed02,zorder=1, color='purple', linewidth=3, linestyle='--')
plt.plot(xSpeed03,ySpeed03,zorder=1, color='purple', linewidth=3, linestyle='--')
plt.plot(xSpeed04,ySpeed04,zorder=1, color='purple', linewidth=3, linestyle='--')
#plt.plot(xSpeed05,ySpeed05,zorder=1, color='purple', linewidth=3, linestyle='--')
plt.plot(xSpeed06,ySpeed06,zorder=1, color='purple', linewidth=3, linestyle='--')
#plt.plot(xSpeed07,ySpeed07,zorder=1, color='purple', linewidth=3, linestyle='--')
#plt.plot(xSpeed08,ySpeed08,zorder=1, color='purple', linewidth=3, linestyle='--')

plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT04,ySpeedGT04,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT05,ySpeedGT05,zorder=1, color='blue', linewidth=3, linestyle='--')
plt.plot(xSpeedGT06,ySpeedGT06,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT07,ySpeedGT07,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT08,ySpeedGT08,zorder=1, color='blue', linewidth=3, linestyle='--')


#plt.plot(xSpeed03,ySpeed03,zorder=1, color='purple', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeed04,ySpeed04,zorder=1, color='purple', linewidth=3, linestyle='--')

#plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT04,ySpeedGT04,zorder=1, color='blue', linewidth=3, linestyle='--')


plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='<', color='purple', linewidth=2)
plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='<', color='purple', linewidth=2)
plt.plot(xSpeed03[-1],ySpeed03[-1],zorder=1, marker='<', color='purple', linewidth=2)
plt.plot(xSpeed04[-1],ySpeed04[-1],zorder=1, marker='<', color='purple', linewidth=2)
plt.plot(xSpeed06[-1],ySpeed06[-1],zorder=1, marker='^', color='purple', linewidth=2)
plt.plot(xSpeedGT01[-1],ySpeedGT01[-1],zorder=1, marker='<', color='blue', linewidth=2)
plt.plot(xSpeedGT02[-1],ySpeedGT02[-1],zorder=1, marker='<', color='blue', linewidth=2)
plt.plot(xSpeedGT03[-1],ySpeedGT03[-1],zorder=1, marker='<', color='blue', linewidth=2)
plt.plot(xSpeedGT04[-1],ySpeedGT04[-1],zorder=1, marker='<', color='blue', linewidth=2)
plt.plot(xSpeedGT06[-1],ySpeedGT06[-1],zorder=1, marker='^', color='blue', linewidth=2)

#plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-5.png")
plt.imshow(img, zorder=0)
plt.show()
