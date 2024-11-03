"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi/
"""

import numpy as np
import cv2 as cv
import utils

image = cv.imread("imgs/einstein.jpg")
cv.imshow("Original Image", image)
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale", image_gray)

# Adding noise to grayscale image
mean = 0
std = 15
noisy_image_gray = utils.add_gauusian_noise(image_gray, mean, std)
cv.imshow("Noisy Image", noisy_image_gray)

# Applying weighted smoothing
weighted_filter = np.array([[1, 1, 1], [1, 3, 1], [1, 1, 1]], dtype=np.float64)
weighted_filter /= np.einsum("ij->", weighted_filter)
size = weighted_filter.shape

weighted_smooth = utils.apply_filter(
    noisy_image_gray, weighted_filter, anchor_filter=(size[0] // 2, size[1] // 2)
)
cv.imshow("Weighted smoothing", weighted_smooth)

# Applying gradient filter to get the edges
edges = utils.gradient(weighted_smooth)
cv.imshow("Edges", edges)

noisy_edges = utils.gradient(noisy_image_gray)
cv.imshow("Edges of noisy image", noisy_edges)


# Saving results
cv.imwrite("imgs/sobel_edge_detector_results/1_original_image.jpg", image)
cv.imwrite("imgs/sobel_edge_detector_results/2_gray_scaled.jpg", image_gray)
cv.imwrite("imgs/sobel_edge_detector_results/3_noisy_image.jpg", noisy_image_gray)
cv.imwrite("imgs/sobel_edge_detector_results/4_weighted_smoothing.jpg", weighted_smooth)
cv.imwrite("imgs/sobel_edge_detector_results/5_noisy_image_edges.jpg", noisy_edges)
cv.imwrite("imgs/sobel_edge_detector_results/6_edges_smooth.jpg", edges)

cv.waitKey(0)
cv.destroyAllWindows()
