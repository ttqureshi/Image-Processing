"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi
"""

import math
import os

import numpy as np
import cv2 as cv


root_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(root_dir, "imgs", "diagonal.png")
img = cv.imread(img_path)
cv.imshow("Lane", img)

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Image", img)


def hough_accumulator_setup(img_edges) -> np.ndarray:
    """
    Parameters
    ----------
    img_edges : np.ndarray
        Image edges

    Returns
    -------
    hough_accumulator : np.ndarray
        Accumulator array

    """
    row, col = img_edges.shape
    diagnoal_length = int(math.sqrt(row**2 + col**2))

    hough_accumulator = np.zeros((180, diagnoal_length))
    return hough_accumulator


def populate_accumulator_array(accumulator, img_space_coords) -> np.ndarray:
    """
    TODO: write doc-string
    """
    x, y = img_space_coords
    theta, two_ro_max = accumulator.shape # here two_ro_max means: `2*ro_max`
    ro_max = accumulator.shape[1] // 2 # `ro_max` is the half-diagonal length

    for theta_not in range(theta):
        ro_prime = int(x * math.cos(theta_not) + y * math.sin(theta_not))

        # We need to MAP from the range [0, 2*ro_max] -> [-ro_max, +ro_max] range
        # if `ro ⋲ [0, 2*ro_max]` AND `ro_prime ⋲ [-ro_max, +ro_max]`
        # then 
        # ``ro_prime = ro - ro_max``
        # ``ro = ro_prime + ro_max``
        ro = ro_prime + ro_max # make sure this is an `INTEGER`
        if ro >= 0 and ro < two_ro_max:
            accumulator[theta_not, ro] += 1

    return accumulator


def hough_transform(img_edges) -> np.ndarray:
    """
    Parameters
    ----------
    img_edges : np.ndarray
        Image edges
    accumulator : np.ndarray
        Accumulator array

    Returns
    -------
    voted_accumulator : np.ndarray #TODO: update it
        An accumulator array with votes indicating potential coordinates for a line in image space.
    """
    n_rows, n_cols = img_edges.shape
    accumulator_array = hough_accumulator_setup(img_edges)
    # voted_accumulator = np.zeros_like(accumulator)

    for r in range(n_rows):
        for c in range(n_cols):
            if img_edges[r, c] == 255:
                coords = (r, c)
                accumulator_array = populate_accumulator_array(accumulator_array, coords)

    return accumulator_array


def sinusoids_in_parameter_space(accumulator_array) -> np.ndarray:
    """
    TODO: write doc-string
    """
    highest_vote = np.max(accumulator_array)
    interval = 255 // highest_vote

    rows, cols = accumulator_array.shape

    for r in range(rows):
        for c in range(cols):
            if accumulator_array[r, c] != 0:
                accumulator_array[r, c] = accumulator_array[r, c] * interval

    return accumulator_array



accumulator = hough_transform(img)
sinusoids = sinusoids_in_parameter_space(accumulator)

cv.imshow("Sinusoids", sinusoids)

write_path = os.path.join(root_dir, "imgs", "sinusoids.png")
cv.imwrite(write_path, sinusoids)


cv.waitKey(0)
cv.destroyAllWindows()

