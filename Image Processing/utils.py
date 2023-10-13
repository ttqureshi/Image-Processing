import numpy as np
import math
import cv2 as cv

def correlation(image_gray, fltr, anchor_filter=[[0,0]], axis=None, pad_zeros=True):
    """

    Parameters
    ----------
    image_gray : Grayscale image
    
    fltr : Filter to be applied on grayscale image
        DESCRIPTION: provide 2D list.
    
    anchor_filter : anchor or pivot of filter, optional
        DESCRIPTION: Default is set to [[0,0]]
    
    axis: The axis along which the filter is to be applied
        DESCRIPTION: only needed when the fltr has one row. 0=>vertical, 1=>horizontal.
                     Default is None (if not provided)
    
    pad_zeros : boolean
        DESCRIPTION: if False, pads the image by mirroring the neighboring pixel values, otherwise pads with 0.
                     The default is True.

    Returns
    -------
    filtered_image : filter applied on image_gray
    
    """
    n_rows = image_gray.shape[0]
    n_cols = image_gray.shape[1]
    
    filtered_image = np.zeros_like(image_gray)
    
    filter_shape = fltr.shape
    if len(filter_shape == 2):
        # 2-D filter
        m = filter_shape[0]
        n = filter_shape[1]
    else:
        # 1-D filter
        m = filter_shape[0]
        n = 0
        
    return filtered_image
















