import numpy as np
import matplotlib.pyplot as plt

xSpeedGT01 = np.asarray([221, 193, 166, 140, 116,  92,  68,  46])
ySpeedGT01 = np.asarray([411, 414, 416, 418, 421, 424, 426, 429])

xSpeed01 = np.asarray([295, 293, 289, 285, 281, 275, 270, 263])
ySpeed01 = np.asarray([260, 257, 255, 252, 250, 248, 245, 243])



plt.plot(xSpeed01,ySpeed01,zorder=1, color='purple', label='0.5', linewidth=3, linestyle='--')
#plt.plot(xSpeed02,ySpeed02,zorder=1, color='purple', linewidth=3, linestyle='--')
plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')


#plt.plot(xSpeed03,ySpeed03,zorder=1, color='purple', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeed04,ySpeed04,zorder=1, color='purple', linewidth=3, linestyle='--')

#plt.plot(xSpeedGT01,ySpeedGT01,zorder=1, color='blue', label='GT', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT02,ySpeedGT02,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT03,ySpeedGT03,zorder=1, color='blue', linewidth=3, linestyle='--')
#plt.plot(xSpeedGT04,ySpeedGT04,zorder=1, color='blue', linewidth=3, linestyle='--')


plt.plot(xSpeed01[-1],ySpeed01[-1],zorder=1, marker='>', color='purple', linewidth=2)
#plt.plot(xSpeed02[-1],ySpeed02[-1],zorder=1, marker='>', color='purple', linewidth=2)
#plt.plot(xSpeedGT01[-1],ySpeedGT01[-1],zorder=1, marker='>', color='blue', linewidth=2)
#plt.plot(xSpeedGT02[-1],ySpeedGT02[-1],zorder=1, marker='>', color='blue', linewidth=2)

plt.legend(loc='center left')
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("../zara1-214.png")
plt.imshow(img, zorder=0)
plt.show()
