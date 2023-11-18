"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi/
"""

import cv2 as cv
import utils

image = cv.imread('imgs/einstein.jpg')
cv.imshow('Original Image', image)
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale',image_gray)

# Adding noise to grayscale image
mean = 0
std = 10
noisy_image_gray = utils.add_gauusian_noise(image_gray, mean, std)
cv.imshow('Noisy Image',noisy_image_gray)

# Applying gaussian smoothing
size = (3,3)
gauss_filter = utils.get_gaussian_dist(size, std=2)

gaussian_smooth = utils.apply_filter(noisy_image_gray, gauss_filter, anchor_filter=(size[0]//2,size[1]//2))
cv.imshow('Gaussian smoothing', gaussian_smooth)

# Applying gradient filter to get the edges
edges = utils.gradient(gaussian_smooth)
cv.imshow('Edges',edges)

noisy_edges = utils.gradient(noisy_image_gray)
cv.imshow('Edges of noisy image',noisy_edges)


# Saving results
cv.imwrite('imgs/dog_filter_results/1_original_image.jpg', image)
cv.imwrite('imgs/dog_filter_results/2_gray_scaled.jpg', image_gray)
cv.imwrite('imgs/dog_filter_results/3_noisy_image.jpg', noisy_image_gray)
cv.imwrite('imgs/dog_filter_results/4_weighted_smoothing.jpg', gaussian_smooth)
cv.imwrite('imgs/dog_filter_results/5_noisy_image_edges.jpg', noisy_edges)
cv.imwrite('imgs/dog_filter_results/6_edges_smooth.jpg', edges)

cv.waitKey(0)
cv.destroyAllWindows()
