import cv2
import numpy as np

bg = cv2.imread("cropblank.png")
for n in range(325 + 1):
    fg = cv2.imread("crop{:03d}.png".format(n))

    diffmask = cv2.absdiff(bg, fg)
    diffmask = cv2.cvtColor(diffmask, cv2.COLOR_RGB2GRAY)
    _, diffmask = cv2.threshold(diffmask, 5, 255, cv2.THRESH_BINARY)

    cv2.imwrite("mask{:03d}.png".format(n), diffmask)

    fg[np.all(diffmask == 0)] = [0, 0, 0]

    cv2.imwrite("masked{:03d}.png".format(n), fg)
