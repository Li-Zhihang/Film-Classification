import cv2.cv2 as cv
import numpy as np

from ..opt import opt
from .hog import myHOG


def get_sym_score(img):

    height = int(img.shape[0])
    width = int(img.shape[1])

    cell_x = height // opt.cell_size[0]
    cell_y = width // opt.cell_size[1]

    # left-right
    img_left = img[:, :width // 2]
    img_right = img[:, width // 2:]

    imgl_rs = np.split(img_left, cell_x, axis=0)
    imgr_rs = np.split(img_right[:, ::-1], cell_x, axis=0)
    histl = np.ndarray((cell_x, cell_y // 2, opt.rgbbins * 3))
    histr = np.ndarray((cell_x, cell_y // 2, opt.rgbbins * 3))
    for r in range(cell_x):
        imgl_rc = np.split(imgl_rs[r], cell_y // 2, axis=1)
        imgr_rc = np.split(imgr_rs[r], cell_y // 2, axis=1)
        for c in range(cell_y // 2):
            histl_r = np.histogram(imgl_rc[c][:, :, 0], opt.rgbbins, (0, 255))[0]
            histl_g = np.histogram(imgl_rc[c][:, :, 1], opt.rgbbins, (0, 255))[0]
            histl_b = np.histogram(imgl_rc[c][:, :, 2], opt.rgbbins, (0, 255))[0]

            histr_r = np.histogram(imgr_rc[c][:, :, 0], opt.rgbbins, (0, 255))[0]
            histr_g = np.histogram(imgr_rc[c][:, :, 1], opt.rgbbins, (0, 255))[0]
            histr_b = np.histogram(imgr_rc[c][:, :, 2], opt.rgbbins, (0, 255))[0]

            histl[r, c] = np.concatenate((histl_r, histl_g, histl_b)) / np.sum(histl_r) / 3
            histr[r, c] = np.concatenate((histr_r, histr_g, histr_b)) / np.sum(histr_r) / 3

    binsl = myHOG(cv.cvtColor(img_left, cv.COLOR_BGR2GRAY), cell_x, cell_y // 2, opt.cell_size)
    binsr = myHOG(cv.cvtColor(img_right[:, ::-1], cv.COLOR_BGR2GRAY), cell_x, cell_y // 2, opt.cell_size)

    cost_lr = np.sum(np.abs(histl - histr)) / cell_x / cell_y / 2
    dist_hoglr = np.sum(np.abs(binsl - binsr)) / cell_x / cell_y / 2

    # top-bottom
    img_top = img[:height // 2]
    img_bottom = img[height // 2:]

    imgt_rs = np.split(img_top, cell_x // 2, axis=0)
    imgb_rs = np.split(img_bottom[::-1], cell_x // 2, axis=0)
    histt = np.ndarray((cell_x // 2, cell_y, opt.rgbbins * 3))
    histb = np.ndarray((cell_x // 2, cell_y, opt.rgbbins * 3))
    for r in range(cell_x // 2):
        imgt_rc = np.split(imgt_rs[r], cell_y, axis=1)
        imgb_rc = np.split(imgb_rs[r], cell_y, axis=1)
        for c in range(cell_y):
            histt_r = np.histogram(imgt_rc[c][:, :, 0], opt.rgbbins, (0, 255))[0]
            histt_g = np.histogram(imgt_rc[c][:, :, 1], opt.rgbbins, (0, 255))[0]
            histt_b = np.histogram(imgt_rc[c][:, :, 2], opt.rgbbins, (0, 255))[0]

            histb_r = np.histogram(imgb_rc[c][:, :, 0], opt.rgbbins, (0, 255))[0]
            histb_g = np.histogram(imgb_rc[c][:, :, 1], opt.rgbbins, (0, 255))[0]
            histb_b = np.histogram(imgb_rc[c][:, :, 2], opt.rgbbins, (0, 255))[0]

            histt[r, c] = np.concatenate((histt_r, histt_g, histt_b)) / np.sum(histt_r) / 3
            histb[r, c] = np.concatenate((histb_r, histb_g, histb_b)) / np.sum(histb_r) / 3

    binst = myHOG(cv.cvtColor(img_top, cv.COLOR_BGR2GRAY), cell_x // 2, cell_y, opt.cell_size)
    binsb = myHOG(cv.cvtColor(img_bottom[::-1], cv.COLOR_BGR2GRAY), cell_x // 2, cell_y, opt.cell_size)

    cost_tb = np.sum(np.abs(histt - histb)) / cell_x / cell_y / 2
    dist_hogtb = np.sum(np.abs(binst - binsb)) / cell_x / cell_y / 2

    return np.array([cost_lr, dist_hoglr, cost_tb, dist_hogtb])
