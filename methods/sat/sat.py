from os.path import join

import numpy as np
from sklearn import svm

try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv


sat_list = ['高饱和', '普通', '消色']
sat_en = ['high_sat', 'normal', 'low_sat']
# parameters
h_bins = 16
s_bins = 8
v_bins = 8
bins_num = h_bins + s_bins + v_bins


def compute_hsv_histograms(imgs):
    datalen = imgs.shape[0]
    hists = np.ndarray((datalen, bins_num))
    for k in range(datalen):
        hsv_img = cv.cvtColor(imgs[k], cv.COLOR_BGR2HSV)
        h_hist = np.histogram(hsv_img[:, :, 0], h_bins, (0, 180))[0]
        h_hist = h_hist / np.sum(h_hist)
        s_hist = np.histogram(hsv_img[:, :, 1], s_bins, (0, 255))[0]
        s_hist = s_hist / np.sum(s_hist)
        v_hist = np.histogram(hsv_img[:, :, 2], v_bins, (0, 255))[0]
        v_hist = v_hist / np.sum(v_hist)
        hists[k] = np.concatenate((h_hist, s_hist, v_hist))

    return hists


def load_dataset(save_dir):
    histograms = np.ndarray((0, bins_num))
    labels = np.ndarray((0), dtype=int)
    for sat_idx in range(len(sat_en)):
        sat = sat_en[sat_idx]
        hists = np.load(join(save_dir, sat + '.npy'))
        histograms = np.concatenate((histograms, hists))
        labels = np.concatenate((labels, sat_idx * np.ones(hists.shape[0], dtype=int)))

    return histograms, labels


class Sat_SVM(object):
    def __init__(self):
        x_std, y_std = load_dataset('./models/sat_std')
        self.svc = svm.SVC(C=1, gamma='scale', verbose=1, max_iter=2000)
        # fit data
        self.svc.fit(x_std, y_std)
        print('Successfully fit Sat SVM')

    def get_sat(self, imgs):

        hists = compute_hsv_histograms(imgs)
        y_preds = self.svc.predict(hists)
        return y_preds
