"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi
"""

import numpy as np
import cv2 as cv
import math

image = cv.imread('imgs/image.jpg')
resized_image = cv.resize(image, (900, 650))
cv.imshow('Image',image)
cv.imshow('Resized Image',resized_image)
image_gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image",image_gray)

# Derivative Filter
der_filter = np.array([-0.5,0,0.5])
anchor = 1

n_rows = image_gray.shape[0]
n_cols = image_gray.shape[1]


filtered_image = np.zeros_like(image_gray)

# Gradient
gradient = np.zeros((2,))
for col in range(n_cols):
    for row in range(n_rows):
        di_dx1 = der_filter[anchor]*image_gray[row,col]
        di_dx2 = der_filter[anchor]*image_gray[row,col]
        for i in range(len(der_filter)):
            if i != anchor:
                traverse = i-anchor
                r = row + traverse
                c = col + traverse
                if r<0:
                    val = 0
                else:
                    try:
                        val = image_gray[r,col]
                    except:
                        val = 0
                di_dx1 += val*der_filter[i]
                
                gradient[0] = abs(di_dx1)
                if c<0:
                    val = 0
                else:
                    try:
                        val = image_gray[row,c]
                    except:
                        val = 0
                di_dx2 += val*der_filter[i]
                gradient[1] = abs(di_dx2)
        mag_gradient = math.sqrt(sum(pow(element, 2) for element in gradient))
        if mag_gradient > 255:
            filtered_image[row,col] = 255
        else:
            filtered_image[row,col] = mag_gradient
        
cv.imshow('Derivative Filter',filtered_image)


cv.waitKey(0)
cv.destroyAllWindows()
