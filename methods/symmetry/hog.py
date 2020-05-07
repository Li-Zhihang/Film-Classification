import cv2.cv2 as cv
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
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[i, j].flatten())
            ang_list = ang_cell[i, j].flatten()
            ang_list = np.int8(ang_list / 20.0)
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])
            bins[i][j] = binn

    return bins


def myHOG(img, cell_x, cell_y, cell_size):

    gradient_values_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14

    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_size)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_size)
    bins = get_bins(grad_cell, ang_cell)

    return bins
