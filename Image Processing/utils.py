"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi/
"""

import numpy as np
import math
import cv2 as cv
import warnings

def apply_filter(image_gray, fltr, anchor_filter=[0,0], pad_zeros=True):
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
    
    di_dx = apply_filter(image_gray, filter_x, anchor_x).astype(np.uint16)
    di_dy = apply_filter(image_gray, filter_y, anchor_y).astype(np.uint16)
    
    gradient_img = np.sqrt(np.square(di_dx) + np.square(di_dy)).astype(np.uint8)
    
    return gradient_img

    

def add_gauusian_noise(image,mean,std):
    """

    Parameters
    ----------
    image : gray scale image
        
    mean : mean of gaussian
        
    std : standard deviation of gaussian
        
    Returns
    -------
    noisy_image: gray scaled noisy image
    """
    noise = np.zeros(image.shape, dtype=np.uint8)
    cv.randn(noise, mean, std)
    noisy_image = cv.add(image, noise)
    return noisy_image


def get_gaussian_dist(size, std):
    """

    Parameters
    ----------
    size : TYPE => tuple
        DESCRIPTION. tuple of size 2 having both entries same and they should be odd not less than (3,3)
    std  : TYPE => int
        DESCRIPTION => standard deviation of gaussian.
        
    Returns
    -------
    gauss_dist: type => ndarray
        DESCRIPTION. gaussian distribution of input size

    """
    x = np.linspace(-1*(size[0]//2), size[0]//2,size[0])
    y = np.linspace(-1*(size[0]//2), size[0]//2,size[0])
    x,y = np.meshgrid(x,y)
    cov_mat = np.identity(2,dtype=np.int8) * std
    det_cov_mat = np.linalg.det(cov_mat)
    inv_cov_mat = np.linalg.inv(cov_mat)
    mean = [0,0]
    constant = 1/(2*np.pi*np.sqrt(det_cov_mat))
    xy_matrix = np.column_stack((x.flatten() - mean[0], y.flatten() - mean[1]))
    exponent = -0.5 * np.einsum('ij,ij->i', np.dot(xy_matrix, inv_cov_mat), xy_matrix)
    gauss_dist = constant * np.exp(exponent).reshape(x.shape)
    return gauss_dist
    
    
    
def edge_detector(image, type):
    """

    Parameters
    ----------
    image : type => ndarray of uint8
        DESCRIPTION. RGB image
    type : string
        DESCRIPTION. tells which edge detector to use. Can only take either "sobel" or "dog" (dog for derivative of gaussian)

    Returns
    -------
    None.

    """
    pass

    



if __name__ == "__main__":
    
# =============================================================================
#     # TEST CASES FOR apply_filter():-
#     image = np.array([
#         [250,35,126,101,41,219,108,4],
#         [143,78,88,234,74,154,27,50],
#         [219,123,230,105,171,55,107,81]
#         ])
#     s = image.shape
#     f = np.array([
#         [-1,1,1],
#         [-1,2,1]
#         ])
#     fs = f.shape
#     anchor = [1,1]
#     
#     result = apply_filter(image,f)
#     print(result)
# =============================================================================
    
    
# =============================================================================
#     # TEST CASES FOR gradient():-
#     image = cv.imread('imgs/mttq.jpg')
#     # resized = cv.resize(image, (300,200))
#     cv.imshow("Image Resized",image)
#     image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     grad = gradient(image_gray)
#     cv.imshow('Edges',grad)
#     
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# =============================================================================
    
    
    # TEST CASES FOR get_gaussian_dist():-
    gauss = get_gaussian_dist((5,5), 1)
    print(gauss)





