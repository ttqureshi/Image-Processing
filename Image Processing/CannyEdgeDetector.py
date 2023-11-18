"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi
"""

import numpy as np
import cv2 as cv
import math
import time
import utils

img = cv.imread('imgs/img.jpg')
cv.imshow('Original Image', img)
# img = cv.resize(img, (950,400))
# cv.imshow('resized', img)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale',img_gray)

# =============================================================================
# Step 1 --- Derivative of Gaussian (DoG) 
# =============================================================================

# first apply the gaussian filter (here i'm using opencv GaussianBlur() function to make it computationally efficient)
gauss_blur = cv.GaussianBlur(img_gray, (3,3), 1)
cv.imshow('Gaussain Blur', gauss_blur)

# now apply the derivative filter (here---using the custom built function, gradient() in utils.py)
dog_edges = utils.gradient(gauss_blur) # here dog_edges means 'edges found by derivative of gaussian'
cv.imshow('DoG', dog_edges)
cv.imwrite('imgs/canny_edge_detector_results/before_non_maxima.jpg', dog_edges)



# =============================================================================
# # Step 2 --- Non-Maxima Suppression
# =============================================================================

# First we need to find the quantized gradient
# v_ks, a numpy array of shape (8,2), where i-th row in v_ks is the direction vector of i-th neighbor w.r.t. the pixel n 
v_ks = np.array([
    [ 0, 1],
    [-1, 1],
    [-1, 0],
    [-1,-1],
    [ 0,-1],
    [ 1,-1],
    [ 1, 0],
    [ 1, 1]
    ])

# unit_v_ks => ndarray of shape same as v_ks, containing the unit direction vectors
def get_vec_mag(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))
def get_unit_vec(vector):
    mag = get_vec_mag(vector)
    return vector/mag

unit_v_ks = np.apply_along_axis(get_unit_vec, axis=1, arr=v_ks)

# non_maxima_arr = np.zeros_like(dog_edges)

# calculating and storing the gradients as well as their magnitude at each pixel of the 'dog_edges' image in a separate arrays 'gradient_vecs' & 'gradient_mags' respectively 
r,c = dog_edges.shape
gradient_vecs = np.zeros((r,c,2), dtype=np.float32)
gradient_mags = np.zeros((r,c), dtype=np.float32)
non_maxima_arr = np.zeros((r,c))

pad_width = ((1,1),(1,1))
padded = np.pad(dog_edges, pad_width, mode='constant', constant_values=0)
m,n = padded.shape

fltr = np.array([-0.5,0,0.5])
gradient = np.zeros((2,))

for i in range(1,m-1):
    for j in range(1,n-1):
        row_slice = padded[i, j-1:j+2]
        col_slice = padded[i-1:i+2, j]
        gradient[0] = np.dot(fltr, col_slice)
        gradient[1] = np.dot(fltr, row_slice)
        gradient_vecs[i-1,j-1] = gradient
        gradient_mags[i-1,j-1] = get_vec_mag(gradient)

# non-maxima suppression algorithm 
def get_quant_grad_dir(r, c, unit_v_ks):
    similarity = np.dot(gradient_vecs[r,c], unit_v_ks.T)/gradient_mags[r,c]
    return np.argmax(similarity)

for i in range(r):
    for j in range(c):
        if (dog_edges[i,j] == 0):
            non_maxima_arr[i,j] = 1 # non-maxima 
        else:
            if (gradient_mags[i,j] != 0):
                k1 = get_quant_grad_dir(i, j, unit_v_ks)
                neighbor_1 = v_ks[(k1+2) % 8] # neighbor of [i,j] in the direction k1 
                neighbor_2 = v_ks[(k1+6) % 8] # neighbor of [i,j] in the opposite direction of k1 
                
                if (gradient_mags[neighbor_1[0], neighbor_1[1]] > gradient_mags[i,j]):
                    non_maxima_arr[i,j] = 1 # mark [i,j] pixel as non-maxima 
                elif (gradient_mags[neighbor_2[0], neighbor_2[1]] > gradient_mags[i,j]):
                    non_maxima_arr[i,j] = 1 # mark [i,j] pixel as non-maxima 
            else:
                try:
                    if (dog_edges[i,j] < dog_edges[i, j+1]):
                        non_maxima_arr[i,j] = 1
                except:
                    pass
                    
                    
# Now use 'non_maxima_arr' as a mask array which is a boolean mask and use it to make the pixels in 'dog_edges' 0 where there's a 1 (non-maxima) in 'non_maxima_arr' 
dog_edges[non_maxima_arr==0] = 0
cv.imshow('Non-maxima suppression', dog_edges)
cv.imwrite('imgs/canny_edge_detector_results/after_non_maxima.jpg', dog_edges)




        

# Now loop over all the pixels and at each pixel calculate the gradient vector and quantize it's direction to any one of the eight direction vectors of it's 8 neighbors     













# =============================================================================
# # Step 3 --- Hysteresis thresholding
# =============================================================================











# =============================================================================

canny = cv.Canny(gauss_blur, 125, 175)
cv.imshow('OpenCV Canny Edge Detector', canny)

cv.waitKey(0)
cv.destroyAllWindows()


