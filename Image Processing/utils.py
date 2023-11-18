"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi/
"""

import numpy as np
import math
import cv2 as cv

def apply_filter(image, fltr, anchor_filter=(0,0), pad_mode='zeros'):
    """
    Parameters
    ----------
    image : image to be filtered
    
    fltr : Filter to be applied on grayscale image 
        DESCRIPTION: takes 2D list 
    
    anchor_filter : TYPE => tuple 
        DESCRIPTION: anchor or pivot of filter, optional 
                     Default is (0,0) 
    
    pad_mode : TYPE => string
        DESCRIPTION: can either be 'zeros' or 'reflect', former pads the image by zeros and latter by mirroring the neighboring pixel values 
                     Default is 'zeros'

    Returns
    -------
    filtered_image : filter applied on image_gray
    
    """
    m, n = fltr.shape
    top_fltr_width = anchor_filter[0]
    bottom_fltr_width = m - anchor_filter[0] - 1
    left_fltr_width = anchor_filter[1]
    right_fltr_width = n - anchor_filter[1] - 1

    pad_width = ((top_fltr_width, bottom_fltr_width),(left_fltr_width, right_fltr_width))
    
    if (pad_mode == 'zeros'):
        padded_img = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)
    elif (pad_mode == 'reflect'):
        padded_img = np.pad(image, pad_width=pad_width, mode='reflect')
    else:
        raise ValueError("Invalid pad mode. Supported modes: 'zeros', 'reflect'")
    
    n_rows, n_cols = padded_img.shape
    filtered_image = np.zeros_like(image)
    
    for i in range(top_fltr_width, n_rows - bottom_fltr_width):
        for j in range(left_fltr_width, n_cols - right_fltr_width):
            overlap = padded_img[(i-top_fltr_width):(i+bottom_fltr_width+1), (j-left_fltr_width):(j+right_fltr_width+1)]
            filtered_image[i-top_fltr_width,j-left_fltr_width] = abs(np.sum(fltr * overlap))
    
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
    anchor_y = (0,1)
    filter_x = filter_y.reshape((-1,1))
    anchor_x = (1,0)
    
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
    


def get_vec_mag(vector):
    """
    Parameters
    ----------
    vector : 1D numpy array

    Returns
    -------
    mag : The magnitude of input vector (1D numpy array)
    """
    mag = math.sqrt(sum(pow(element, 2) for element in vector))
    return mag




def get_unit_vec(vector):
    """
    Uses the get_vec_mag() function and returns the unit vector
    """
    mag = get_vec_mag(vector)
    return vector/mag 


    
# =============================================================================
# def edge_detector(image, size, mode='dog', std=10):
#     """
# 
#     Parameters
#     ----------
#     image : TYPE => ndarray of uint8
#         DESCRIPTION. RGB image
#     size : TYPE => tuple
#         DESCRIPTION. size of kernel/filter
#     mode : TYPE -> string, optional
#         DESCRIPTION. The default is 'dog' (derivative of gaussian). mode can only be "sobel" or "dog"
#     std : TYPE => int, optional
#         DESCRIPTION. Only needed in case of 'DoG' filter to set the standard deviation of gaussian.
# 
#     Returns
#     -------
#     edges : TYPE -> ndarray of uint8
#         DESCRIPTION. returns the image showing the edges of input image
# 
#     """
#     image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     match mode:
#         case "dog":
#             gauss_filter = get_gaussian_dist(size, std)
#             smoothing = apply_filter(image_gray, gauss_filter, anchor_filter=(size[0]//2,size[1]//2))
#             edges = gradient(smoothing)
#         case "sobel":
#             np
# =============================================================================
    

    



if __name__ == "__main__":
    
    # TEST CASES FOR apply_filter():-
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
    anchor = (1,1)
    
    result = apply_filter(image,f,anchor_filter=anchor, pad_mode='reflect')
    print(result)

    
    
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





