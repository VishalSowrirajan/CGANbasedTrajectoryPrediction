import numpy as np
import matplotlib.pyplot as plt

#### WITHOUT POOLING MODULE ####

#xSpeed01 = np.asarray([344, 322, 299, 278])
#ySpeed01 = np.asarray([365, 368, 370, 371])

#xSpeed02 = np.asarray([355, 333, 310, 288])
#ySpeed02 = np.asarray([450, 447, 444, 440])

#xSpeed03 = np.asarray([357, 337, 314, 293])
#ySpeed03 = np.asarray([411, 409, 405, 402])

#xSpeed04 = np.asarray([239, 263, 288, 311])
#ySpeed04 = np.asarray([376, 365, 356, 348])

#### WITH POOLING MODULE ####

#xSpeed01 = np.asarray([345, 325, 303, 281])
#ySpeed01 = np.asarray([364, 367, 368, 370])

#xSpeed02 = np.asarray([353, 330, 306, 282])
#ySpeed02 = np.asarray([450, 448, 446, 445])

#xSpeed03 = np.asarray([357, 336, 313, 290])
#ySpeed03 = np.asarray([411, 409, 407, 405])

#xSpeed04 = np.asarray([240, 266, 292, 317])
#ySpeed04 = np.asarray([383, 382, 383, 384])

### Ground Truth ###

#xSpeed01 = np.asarray([344, 322, 301, 280])
#ySpeed01 = np.asarray([370, 378, 383, 385])

#xSpeed02 = np.asarray([351, 327, 305, 282])
#ySpeed02 = np.asarray([452, 451, 450, 450])

#xSpeed03 = np.asarray([354, 330, 308, 285])
#ySpeed03 = np.asarray([412, 411, 412, 413])

#xSpeed04 = np.asarray([239, 264, 290, 316])
#ySpeed04 = np.asarray([377, 369, 365, 363])

### SIMULATED WITH SPEED 0.6 ###

xSpeed01 = np.asarray([324, 302, 281, 259])
ySpeed01 = np.asarray([373, 376, 376, 374])

xSpeed02 = np.asarray([332, 312, 292, 272])
ySpeed02 = np.asarray([447, 442, 436, 428])

xSpeed03 = np.asarray([335, 315, 295, 275])
ySpeed03 = np.asarray([408, 404, 398, 390])

xSpeed04 = np.asarray([260, 281, 304, 327])
ySpeed04 = np.asarray([367, 357, 346, 336])
#plt.scatter(xgt,ygt,label="GT")
plt.plot(xSpeed01,ySpeed01,zorder=1)
plt.plot(xSpeed02,ySpeed02,zorder=1)
plt.plot(xSpeed03,ySpeed03,zorder=1)
plt.plot(xSpeed04,ySpeed04,zorder=1)

#plt.legend()
plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("48second.png")
plt.imshow(img, zorder=0)
plt.show()
