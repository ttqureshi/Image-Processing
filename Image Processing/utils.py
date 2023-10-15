"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi/
"""

import numpy as np
import math
import cv2 as cv

def correlation(image_gray, fltr, anchor_filter=[0,0], pad_zeros=True):
    """

    Parameters
    ----------
    image_gray : Grayscale image
    
    fltr : Filter to be applied on grayscale image
        DESCRIPTION: takes 2D list.
    
    anchor_filter : anchor or pivot of filter, optional
        DESCRIPTION: Default is set to [0,0].
    
    pad_zeros : boolean
        DESCRIPTION: if False, pads the image by mirroring the neighboring pixel values, otherwise pads with 0.
                     Default is True.

    Returns
    -------
    filtered_image : filter applied on image_gray
    
    """
    fltr_shape = fltr.shape
    m = fltr_shape[0]
    n = fltr_shape[1]
    
    n_rows = image_gray.shape[0]
    n_cols = image_gray.shape[1]
    
    filtered_image = np.zeros_like(image_gray)
    for col in range(n_cols):
        for row in range(n_rows):
            mul_accumulate = 0
            for rf in range(m):
                traverse_row = rf-anchor_filter[0]
                r = row + traverse_row
                for cf in range(n):
                    traverse_col = cf-anchor_filter[1]
                    c = col + traverse_col
                    
                    if r<0 or c<0:
                        # image is being padded with zero
                        val = 0
                    else:
                        try:
                            val = image_gray[r,c]
                        except:
                            val = 0
                    mul_accumulate += val*fltr[rf,cf]
            filtered_image[row,col] = abs(mul_accumulate)
    return filtered_image


def gradient(image_gray):
    """

    Parameters
    ----------
    image_gray : Grayscale image
        DESCRIPTION.

    Returns
    -------
    gradient_img: edges detected in image_gray

    """
    filter_y = np.array([
        [-0.5,0,0.5]
        ])
    anchor_y = [0,1]
    filter_x = filter_y.reshape((-1,1))
    anchor_x = [1,0]
    
    di_dx = correlation(image_gray, filter_x, anchor_x).astype(np.uint16)
    di_dy = correlation(image_gray, filter_y, anchor_y).astype(np.uint16)
    
    gradient_img = np.sqrt(np.square(di_dx) + np.square(di_dy)).astype(np.uint8)
    
    return gradient_img
        



if __name__ == "__main__":
    # TEST CASES FOR correlation():-
    
    image = np.array([
        [250,35,126,101,41,219,108,4],
        [143,78,88,234,74,154,27,50],
        [219,123,230,105,171,55,107,81]
        ])
    s = image.shape
    f = np.array([
        [-1,1,1],
        [-1,2,1]
        ])
    fs = f.shape
    
    result = correlation(image,f,[1,1])
    print(result)
    
    
    # TEST CASES FOR gradient():-
    image = cv.imread('imgs/image.jpg')
    resized = cv.resize(image, (900,650))
    cv.imshow("Image Resized",resized)
    image_gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    grad = gradient(image_gray)
    cv.imshow('Edges',grad)
    
    cv.waitKey(0)
    cv.destroyAllWindows()





