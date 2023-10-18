"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi/
"""

import numpy as np
import math
import cv2 as cv
import warnings

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
    size : TYPE => ndarray
        DESCRIPTION. a vector to specify the mean position. mean should have odd values i.e., [3,3], [5,5] etc. If even is given rounds it to the nearest odd value tuple.
    std  : TYPE => int
        DESCRIPTION => standard deviation of gaussian.

    Returns
    -------
    gauss_dist: type => ndarray
        DESCRIPTION. gaussian distribution of input size

    """
    def gaussian_3d(n,mean,cov_mat):
        """

        Parameters
        ----------
        n : TYPE => ndarray
            DESCRIPTION => a vector to specify the point where we want to get the gaussian distribution value
        mean : TYPE => ndarray
            DESCRIPTION => mean of gaussian distribution. A vector with 2 elements to specify x and y coordinates of mean 
        cov_mat : TYPE => ndarray
            DESCRIPTION => a 2D diagonal matrix with diagnoal entries equal to standard deviation of distribution

        Returns
        -------
        gauss_n: TYPE => float
            DESCRIPTION => value of gaussian at index [x,y]

        """
        det_cov_mat = np.linalg.det(cov_mat)
        inv_cov_mat = np.linalg.inv(cov_mat)
        
        constant = 1/(2*np.pi*np.sqrt(det_cov_mat))
        mean_dist = n - mean # distance of point n from mean
        exponent = np.exp(-0.5*np.transpose(mean_dist)*np.square(inv_cov_mat)*mean_dist)
        
        gauss_n = constant*exponent.flatten()[0]
        return gauss_n
    
    mean = (size//2).reshape(1,-1)
    cov_mat = np.identity(2,dtype=np.int8) * std
    gauss_dist = np.zeros(size)
    
    for x in range(size[0]):
        for y in range(size[1]):
            n = np.array([x,y])
            val_n = gaussian_3d(n, mean, cov_mat)
            gauss_dist[n[0],n[1]] = val_n
    return gauss_dist
    
    
    
    
def denoise_gaussian(image, ker_size=(3,3)):
    """
    Applies the gaussian filter to the noisy image and returns the smoothened or denoised image


    Parameters
    ----------
    image : noisy image
        
    ker_size : size of filter, takes a tuple, optional
        DESCRIPTION. The default is (3,3).
                     Kernel size can only take odd values if not provided then takes the nearest odd size.

    Returns
    -------
    denoised_img: denoised image

    """
    pass
    
    



if __name__ == "__main__":
    
# =============================================================================
#     # TEST CASES FOR correlation():-
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
#     result = correlation(image,f)
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
    gauss = get_gaussian_dist(np.array([5,5]), 10)
    print(gauss)





