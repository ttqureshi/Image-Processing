import numpy as np
import cv2 as cv
import utils

image = cv.imread('imgs/person.jpg')
cv.imshow('Original Image',image)
g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
temp = utils.gradient(g)
cv.imshow('before adding noise',temp)

# Adding noise to the image
mean = 0
stddev = 70
# noise = np.zeros(image.shape, dtype=np.uint8)
# cv.randn(noise,mean,stddev)
# noisy_img = cv.add(image,noise)
# cv.imshow('noisy Image',noisy_img)

noisy_image = utils.add_gauusian_noise(image, mean, stddev)
cv.imshow('noisy Image',noisy_image)

weighted_filter = np.array([
    [1/11,1/11,1/11],
    [1/11,3/11,1/11],
    [1/11,1/11,1/11]
    ])

filtered = utils.correlation(noisy_image, weighted_filter, [1,1])
cv.imshow('filtered',filtered)
cv.imwrite('imgs/denoised.jpg', filtered)

image_gray = cv.cvtColor(noisy_image, cv.COLOR_BGR2GRAY)

edges = utils.gradient(image_gray)
cv.imshow('edges',edges)

cv.waitKey(0)
cv.destroyAllWindows()


