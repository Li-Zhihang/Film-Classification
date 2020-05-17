import argparse
import os

import cv2.cv2 as cv
import numpy as np


tone_list = ['全色调', '白灰黑低调', '灰黑低调', '白黑低调', '白灰黑高调', '白灰高调', '白黑高调', '低反差']
tone_en = ['full_color', 'wgb_low', 'gb_low', 'wb_low', 'wgb_high', 'wg_high', 'wb_high', 'low_contrast']
# parameters
h_bins = 16
s_bins = 8
v_bins = 8
bins_num = h_bins + s_bins + v_bins
gray_bins = 128


def compute_hsv_histograms(path, shape):

    img = cv.imread(path)
    img = cv.resize(img, (shape[1], shape[0]))  # resize for faster processing

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h_hist = np.histogram(hsv_img[0], h_bins, (0, 180))[0]
    h_hist = h_hist / np.sum(h_hist)
    s_hist = np.histogram(hsv_img[1], s_bins, (0, 255))[0]
    s_hist = s_hist / np.sum(s_hist)
    v_hist = np.histogram(hsv_img[2], v_bins, (0, 255))[0]
    v_hist = v_hist / np.sum(v_hist)
    hist = np.concatenate((h_hist, s_hist, v_hist))

    return hist


def compute_gray_histograms(path, shape):
    img = cv.imread(path)
    img = cv.resize(img, (shape[1], shape[0]))  # resize for faster processing

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = np.histogram(hsv_img, gray_bins, (0, 256))[0]
    hist = hist / np.sum(hist)

    return hist


def main(args):
    for tone_idx in range(len(tone_list)):
        print('Proceeding ' + tone_list[tone_idx] + '...   ', end='')
        folder_path = os.path.join(args.tgt, tone_list[tone_idx])
        img_list = os.listdir(folder_path)

        # hists = np.ndarray((len(img_list), bins_num))
        hists = np.ndarray((len(img_list), gray_bins))

        for f_idx in range(len(img_list)):
            f = img_list[f_idx]
            path = os.path.join(folder_path, f)

            # hists[f_idx, :] = compute_hsv_histograms(path)
            hists[f_idx, :] = compute_gray_histograms(path, args.shape)

        # split into training and testing
        np.random.shuffle(hists)
        test_len = len(hists) // 6
        hists_test = hists[-test_len:]
        hists_train = hists[:-test_len]

        np.save(os.path.join(args.save_dir, 'train', tone_en[tone_idx] + '.npy'), hists_train)
        np.save(os.path.join(args.save_dir, 'test', tone_en[tone_idx] + '.npy'), hists_test)
        print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt', type=str, default='/media/li-zhihang/LiZhihang/摄影标签细分/影调')
    parser.add_argument('--save_dir', type=str, default='./dataset/tone/gray')
    parser.add_argument('--shape', type=tuple, default=(360, 640))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'train'), exist_ok=True)

    main(args)
