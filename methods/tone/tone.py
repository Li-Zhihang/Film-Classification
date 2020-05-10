try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv

import keras
import numpy as np

from ..opt import opt

tone_list = ['全色调', '白灰黑低调', '灰黑低调', '白黑低调', '白灰黑高调', '白灰高调', '白黑高调', '低反差']
tone_en = ['full_color', 'wgb_low', 'gb_low', 'wb_low', 'wgb_high', 'wg_high', 'wb_high', 'low_contrast']  # , 'high_contrast']


def perceptron(input_dim):
    net = keras.models.Sequential([
        keras.layers.Dense(200, input_shape=(input_dim,), activation='relu'),
        keras.layers.Dense(len(tone_en), activation='softmax')
    ])
    return net


def compute_gray_histograms(imgs):

    gray_hists = np.ndarray((imgs.shape[0], opt.gray_bins))
    for n in range(imgs.shape[0]):
        hsv_img = cv.cvtColor(imgs[n], cv.COLOR_BGR2GRAY)
        hist = np.histogram(hsv_img, opt.gray_bins, (0, 256))[0]
        gray_hists[n] = hist / np.sum(hist)

    return gray_hists


class ToneClassifier(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = perceptron(opt.gray_bins)
        self.model.load_weights('./models/tone_mlp/model_21.hdf5')
        print('Successfully load ToneClassifier')

    def get_tone_class(self, imgs):
        datalen = imgs.shape[0]
        gray_hists = compute_gray_histograms(imgs)

        # run inference
        leftover = not (datalen % self.batch_size == 0)
        batch_num = datalen // self.batch_size + leftover

        tone_type = np.ndarray((datalen,), dtype=int)
        for batch_idx in range(batch_num):
            start = batch_idx * self.batch_size
            stop = min((batch_idx + 1) * self.batch_size, datalen)
            preds = self.model.predict(gray_hists[start:stop])
            preds_type = np.argmax(preds, axis=-1)
            tone_type[start:stop] = preds_type

        return tone_type
