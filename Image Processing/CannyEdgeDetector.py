import numpy as np
import cv2 as cv
import utils

img = cv.imread('imgs/einstein.jpg')
cv.imshow('Original Image', img)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale',img_gray)

# =============================================================================
# Step 1 --- Derivative of Gaussian (DoG) 
# =============================================================================

# first apply the gaussian filter (here i'm using opencv GaussianBlur() function to make it computationally efficient)
gauss_blur = cv.GaussianBlur(img_gray, (5,5), 1)
cv.imshow('Gaussain Blur', gauss_blur)

# now apply the derivative filter (here---using the custom built function, gradient() in utils.py)
dog_edges = utils.gradient(gauss_blur) # here dog_edges means 'edges found by derivative of gaussian'
cv.imshow('DoG', dog_edges)



# =============================================================================
# # Step 2 --- Non-Maxima Suppression
# =============================================================================

# First we need to find the quantized gradient
# Writing the direction vectors (v_1 to v_8) pointing in the direction of eight neighbors given a pixel n.   
v_1 = np.array([ 0, 1])
v_2 = np.array([-1, 1])
v_3 = np.array([-1, 0])
v_4 = np.array([-1,-1])
v_5 = np.array([ 0,-1])
v_6 = np.array([ 1,-1])
v_7 = np.array([ 1, 0])
v_8 = np.array([ 1, 1])

# Now loop over all the pixels and at each pixel calculate the gradient vector and quantize it's direction to any one of the eight direction vectors of it's 8 neighbors     













# =============================================================================
# # Step 3 --- Hysteresis thresholding
# =============================================================================











# =============================================================================

canny = cv.Canny(gauss_blur, 125, 175)
cv.imshow('OpenCV Canny Edge Detector', canny)

cv.waitKey(0)
cv.destroyAllWindows()


