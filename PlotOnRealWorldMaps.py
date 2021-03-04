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

#xSpeed01 = np.asarray([324, 302, 281, 259])
#ySpeed01 = np.asarray([373, 376, 376, 374])

#xSpeed02 = np.asarray([332, 312, 292, 272])
#ySpeed02 = np.asarray([447, 442, 436, 428])

#xSpeed03 = np.asarray([335, 315, 295, 275])
#ySpeed03 = np.asarray([408, 404, 398, 390])

#xSpeed04 = np.asarray([260, 281, 304, 327])
#ySpeed04 = np.asarray([367, 357, 346, 336])
#plt.scatter(xgt,ygt,label="GT")
#plt.plot(xSpeed01,ySpeed01,zorder=1, marker='.')
#plt.plot(xSpeed02,ySpeed02,zorder=1, marker='.')
#plt.plot(xSpeed03,ySpeed03,zorder=1, marker='.')
#plt.plot(xSpeed04,ySpeed04,zorder=1, marker='.')
#img = plt.imread("48second.png")

#### PLOTTING ZARA-1 WITH 25th SECOND SIMULATED 0.8 ####
#xSpeed01 = np.asarray([192, 170, 146, 122, 96])
#ySpeed01 = np.asarray([346, 344, 342, 339, 336])

#xSpeed02 = np.asarray([112, 124, 136, 149, 160])
#ySpeed02 = np.asarray([278, 268, 257, 244, 230])

#xSpeed03 = np.asarray([129, 140, 153, 167])
#ySpeed03 = np.asarray([291, 285, 277, 268])

#xSpeed04 = np.asarray([104, 119, 136, 155, 175])
#ySpeed04 = np.asarray([323, 320, 315, 309, 300])

#xSpeed05 = np.asarray([105])
#ySpeed05 = np.asarray([326])

#plt.plot(xSpeed01,ySpeed01,zorder=1, marker='<')
#plt.plot(xSpeed02,ySpeed02,zorder=1, marker='>')
#plt.plot(xSpeed03,ySpeed03,zorder=1, marker='>')
#plt.plot(xSpeed04,ySpeed04,zorder=1, marker='>')
#plt.scatter(xSpeed05,ySpeed05,zorder=1)

# PLOTTING HOTEL DATASET - 23rd second

#xSpeed05 = np.asarray([456, 435, 416, 397, 379, 360, 342, 323])
#ySpeed05 = np.asarray([335, 334, 333, 334, 336, 339, 342, 345])

#xSpeed06 = np.asarray([467, 448, 430, 412, 393, 375, 358, 340])
#ySpeed06 = np.asarray([291, 291, 292, 294, 297, 301, 305, 308])

#plt.plot(xSpeed05, ySpeed05, zorder=1, marker='.')
#plt.plot(xSpeed06, ySpeed06, zorder=1, marker='.')

# GROUP AVOIDANCE PLOT - PLOTTING HOTEL DATASET - 114 second

#xSpeed05 = np.asarray([263, 237, 216, 196, 178, 161, 145, 130])
#ySpeed05 = np.asarray([384, 371, 360, 352, 347, 342, 338, 334])

#xSpeed06 = np.asarray([265, 240, 218, 199, 180, 162, 146, 130])
#ySpeed06 = np.asarray([344, 332, 323, 317, 313, 309, 307, 304])

#plt.plot(xSpeed05, ySpeed05, zorder=1, marker='<', color='red', linewidth=2)
#plt.plot(xSpeed06, ySpeed06, zorder=1, marker='<', color='yellow', linewidth=2)
#img = plt.imread("hotel115.png")
#img = plt.imread("hotel382.png")

### PLOTTING GROUP AVOIDANCE WITH SPEED REGRESSOR DATA ###

#plt.legend()

# PLOTTING HOTEL DATASET - Plot 1 in figure 3

#xSpeed01 = np.asarray([447, 448, 448])
#ySpeed01 = np.asarray([227, 226, 227])

#xSpeed02 = np.asarray([289, 267, 244, 222, 198])
#ySpeed02 = np.asarray([322, 321, 321, 322, 323])

#xSpeed03 = np.asarray([424, 393, 363, 335, 310])
#ySpeed03 = np.asarray([348, 333, 320, 308, 297])

#xSpeed04 = np.asarray([440, 406, 374, 345, 318])
#ySpeed04 = np.asarray([293, 278, 265, 254, 244])

#xSpeed05 = np.asarray([207, 238, 264, 291, 317])
#ySpeed05 = np.asarray([395, 395, 393, 390, 387])

#xSpeed06 = np.asarray([198, 226, 251, 276, 299])
#ySpeed06 = np.asarray([424, 422, 419, 416, 411])

#xSpeed07 = np.asarray([563, 532, 503, 475, 450])
#ySpeed07 = np.asarray([357, 342, 331, 320, 311])

#plt.scatter(xSpeed01, ySpeed01, zorder=1)
#plt.plot(xSpeed02, ySpeed02, zorder=1, marker='<', color='yellow')
#plt.scatter(xSpeed03, ySpeed03, zorder=1)
#plt.scatter(xSpeed04, ySpeed04, zorder=1)
#plt.plot(xSpeed05, ySpeed05, zorder=1, marker='>', color='indigo')
#plt.plot(xSpeed06, ySpeed06, zorder=1, marker='>', color='red')
#plt.scatter(xSpeed07, ySpeed07, zorder=1)

#plt.plot(xSpeed06, ySpeed06, zorder=1, marker='.', linestyle='--', color='blue')
#plt.plot(xSpeed05_gt, ySpeed05_gt, zorder=1, marker='.', color='magenta')
#plt.plot(xSpeed06_gt, ySpeed06_gt, zorder=1, marker='.', color='magenta')
#plt.plot(xSpeed06, ySpeed06, zorder=1, marker='.')

plt.xticks([])
plt.yticks([])
ext = [0, 576, 0.00, 720]
img = plt.imread("hotel382.png")
plt.imshow(img, zorder=0)
plt.show()
