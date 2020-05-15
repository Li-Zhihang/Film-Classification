import argparse
import os

import face_recognition as fr
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import layers, models

try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
pos_list_en = ['sl', 'bsl', 'fsl', 'fol', 'tl', 'frl']
pos_list = ['侧光', '侧逆光', '前侧光', '脚光', '顶光', '顺光']
# above abbrs stand for: sidelight, back-sidelight, front-sidelight,
# footlight, backlight, toplight, frontlight


def LenetBuilder():
    net = models.Sequential([
        layers.Conv2D(10, 5, activation='relu', input_shape=(3, 224, 224)),
        layers.MaxPool2D(strides=2),
        layers.Conv2D(10, 5, activation='relu'),
        layers.MaxPool2D(strides=2),
        layers.Conv2D(10, 6, activation='relu'),
        layers.MaxPool2D(strides=2),
        layers.Conv2D(20, 5, activation='relu'),
        layers.MaxPool2D(strides=2),
        layers.Conv2D(50, 3, activation='relu'),
        layers.MaxPool2D(strides=2),
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])
    return net


def max_area_indx(locations):
    '''return index of boundbox regard to the max area'''
    num = len(locations)
    if num == 1:
        return 0
    else:
        max_area = 0
        for indx in range(num):
            box = locations[indx]
            area = (box[2] - box[0]) * (box[1] - box[3])
            if area > max_area:
                max_indx = indx
                max_area = area
        return max_indx


def crop_face(image):
    # find faces
    locations = fr.api.face_locations(image, model='cnn')

    if not locations:
        print('\033[33m[WARN] Cannot find face in file.\033[37m')
        return None
    flags = locations[max_area_indx(locations)]

    cropped = cv.resize(image[flags[0]:flags[2], flags[3]:flags[1], :], (224, 224))
    return cropped


def read_picture(path, model):
    print('Reading ' + path)
    image = cv.imread(path)  # [h, w, c]

    expo = pow(image.shape[0] * image.shape[1] / 500000, 0.5)
    image = cv.resize(image, (int(image.shape[1] / expo), int(image.shape[0] / expo)))

    cropped = crop_face(image)

    if cropped is not None:
        cropped = cropped[np.newaxis, :, :, :]
        cropped = np.transpose(cropped, (0, 3, 1, 2))
        pred = model.predict(cropped, verbose=0)
        for x in range(len(pos_list)):
            print(pos_list[x] + ': {:2f}'.format(pred[0, x]))


def main_infer(args):
    K.set_image_data_format('channels_first')
    model = LenetBuilder()
    model.load_weights(args.ckpt_path)

    if os.path.isdir(args.tgt):
        for file in os.listdir(args.tgt):
            read_picture(os.path.join(args.tgt, file), model)
    else:
        read_picture(args.tgt, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, default='./Records/model_04.hdf5')
    parser.add_argument('--tgt', type=str, default='./dataset/test_light')

    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError('Ckpt path does not exist.')

    if not os.path.exists(args.tgt):
        raise FileNotFoundError('Target path does not exist.')
    main_infer(args)
