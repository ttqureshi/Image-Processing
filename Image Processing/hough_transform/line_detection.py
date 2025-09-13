"""
Author: Muhammad Tayyab Tahir Qureshi
Github: github.com/ttqureshi
"""
import math
import os

import numpy as np
import cv2 as cv


root_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(root_dir, "imgs", "lane.jpg")
img = cv.imread(img_path)
cv.imshow("Lane", img)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Lane Gray", img_gray)

img_edges = cv.Canny(img_gray, 300, 460)
cv.imshow("Lane Edges", img_edges)


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
    theta, ro = accumulator.shape
    mid_diagonal =  accumulator.shape[0] // 2

    for theta_not in range(theta):
        ro_not = int(x * math.cos(theta_not) + y * math.sin(theta_not))
        # ro_not = ro_not + mid_diagonal # confusing (it might work), if anything breaks come here
        ro_not = ro_not // 2

        if abs(ro_not) <= mid_diagonal:
            accumulator[theta_not, ro_not] += 1

    return


def hough_transform(img_edges, accumulator) -> np.ndarray:
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
    # voted_accumulator = np.zeros_like(accumulator)

    for r in range(n_rows):
        for c in range(n_cols):
            if img_edges[r, c] == 255:
                # call `populate_accumulator_array()`
                pass

    return



accumulator = hough_accumulator_setup(img_edges)


cv.waitKey(0)
cv.destroyAllWindows()

