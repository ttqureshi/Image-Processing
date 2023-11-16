"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi
"""

import numpy as np
import cv2 as cv
import time
import math

image = cv.imread('imgs/image.jpg')
resized_image = cv.resize(image, (900, 650))
cv.imshow('Image',image)
cv.imshow('Resized Image',resized_image)
image_gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image",image_gray)



# Efficient version

pad_width = ((1,1),(1,1))
padded = np.pad(image_gray, pad_width, mode='constant', constant_values=0)

m,n = padded.shape

fltr = np.array([-0.5,0,0.5])
new_image = np.zeros_like(image_gray)
def get_vec_mag(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))
gradient = np.zeros((2,))
for i in range(1,m-1):
    for j in range(1,n-1):
        row_slice = padded[i, j-1:j+2]
        col_slice = padded[i-1:i+2, j]
        gradient[0] = np.dot(fltr, col_slice)
        gradient[1] = np.dot(fltr, row_slice)
        
        mag = get_vec_mag(gradient)
        if mag > 255:
            new_image[i-1,j-1] = 255
        else:
            new_image[i-1,j-1] = mag

        





# =============================================================================
# # OLD IMPLEMENTATION
# t1 = time.time()
# der_filter = np.array([-0.5,0,0.5])
# anchor = 1
# 
# n_rows = image_gray.shape[0]
# n_cols = image_gray.shape[1]
# 
# 
# filtered_image = np.zeros_like(image_gray)
# 
# # Gradient
# gradient = np.zeros((2,))
# for col in range(n_cols):
#     for row in range(n_rows):
#         di_dx1 = der_filter[anchor]*image_gray[row,col]
#         di_dx2 = der_filter[anchor]*image_gray[row,col]
#         for i in range(len(der_filter)):
#             if i != anchor:
#                 traverse = i-anchor
#                 r = row + traverse
#                 c = col + traverse
#                 if r<0:
#                     val = 0
#                 else:
#                     try:
#                         val = image_gray[r,col]
#                     except:
#                         val = 0
#                 di_dx1 += val*der_filter[i]
#                 
#                 gradient[0] = (di_dx1)
#                 if c<0:
#                     val = 0
#                 else:
#                     try:
#                         val = image_gray[row,c]
#                     except:
#                         val = 0
#                 di_dx2 += val*der_filter[i]
#                 gradient[1] = (di_dx2)
#         mag_gradient = math.sqrt(sum(pow(element, 2) for element in gradient))
#         if mag_gradient > 255:
#             filtered_image[row,col] = 255
#         else:
#             filtered_image[row,col] = mag_gradient
#         
# print(time.time()-t1)
# cv.imshow('Derivative Filter',filtered_image)
#         
# =============================================================================


cv.imshow('Derivative', new_image)


cv.waitKey(0)
cv.destroyAllWindows()
