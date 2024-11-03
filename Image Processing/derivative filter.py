"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi
"""

import numpy as np
import cv2 as cv
import time
import math

image = cv.imread("imgs/image.jpg")
resized_image = cv.resize(image, (900, 650))
cv.imshow("Image", image)
cv.imshow("Resized Image", resized_image)
image_gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", image_gray)


pad_width = ((1, 1), (1, 1))
padded = np.pad(image_gray, pad_width, mode="constant", constant_values=0)

m, n = padded.shape

fltr = np.array([-0.5, 0, 0.5])
new_image = np.zeros_like(image_gray)


def get_vec_mag(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


gradient = np.zeros((2,))
for i in range(1, m - 1):
    for j in range(1, n - 1):
        i_axis_slice = padded[i - 1 : i + 2, j]
        j_axis_slice = padded[i, j - 1 : j + 2]
        gradient[0] = np.dot(fltr, i_axis_slice)
        gradient[1] = np.dot(fltr, j_axis_slice)

        mag = get_vec_mag(gradient)
        if mag > 255:
            new_image[i - 1, j - 1] = 255
        else:
            new_image[i - 1, j - 1] = mag

cv.imshow("Derivative", new_image)

cv.waitKey(0)
cv.destroyAllWindows()
