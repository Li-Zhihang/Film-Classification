import cv2 as cv
import numpy as np


def gamma(img):
    return np.power(img / 255.0, 1)


def div(img, cell_x, cell_y, cell_size):
    cell = np.zeros(shape=(cell_x, cell_y, cell_size[0], cell_size[1]))
    img_x = np.split(img, cell_x, axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell


def get_bins(grad_cell, ang_cell):
    bins = np.zeros((10))
    grad_cell = grad_cell.flatten()
    ang_cell = ang_cell.flatten()
    bidx = 1
    for bidx in range(1, 10):
        bin_top = -np.pi + 2 * bidx * np.pi / 9
        bin_count = ang_cell < bin_top
        bins[bidx] = np.sum(grad_cell[bin_count])

    res = bins[1:] - bins[0:-1]
    return res / (np.linalg.norm(res) + 0.01)


def myHOG(img, cell_x, cell_y, cell_size):

    gradient_values_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)

    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_size)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_size)
    bins = np.zeros((cell_x, cell_y, 9))
    for r in range(cell_x):
        for c in range(cell_y):
            bins[r, c] = get_bins(grad_cell[r, c], ang_cell[r, c])

    return bins
