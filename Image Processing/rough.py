# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:40:24 2023

@author: T460
"""
# import numpy as np

# # Assuming you have an n x n NumPy array
# n = 3  # Replace with the size of your array
# matrix = np.random.random((n, n))

# # Split the array into column vectors
# column_vectors = np.hsplit(matrix, n)

# # Display the original matrix and the column vectors
# print("Original Matrix:")
# print(matrix)

# print("\nColumn Vectors:")
# for i, col_vector in enumerate(column_vectors):
#     print(f"Column {i + 1}:", col_vector)



# import numpy as np
# import cv2 as cv

# image = cv.imread('imgs/image.jpg')
# image_shape = image.shape
# cv.imshow("Original Image",image)
# image[:,:,0] = np.zeros_like(image[:,:,0])
# image[:,:,1] = np.zeros_like(image[:,:,1])
# r_channel = image[:,:,2]
# r_shape = r_channel.shape
# cv.imshow('r_channel',r_channel)
# cv.imshow('red Channel',image)

# cv.waitKey(0)
# cv.destroyAllWindows()



import numpy as np

a = np.array([[0,1]])
shape = a.shape
aa = a[0].shape









