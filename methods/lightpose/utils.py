try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv

import face_recognition as fr
import numpy as np
from keras import layers, models


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
        return None
    flags = locations[max_area_indx(locations)]

    cropped = cv.resize(image[flags[0]:flags[2], flags[3]:flags[1], :], (224, 224))
    return cropped


def numpy_convert_to_list(array):
    list_array = []
    for idx in range(array.shape[0]):
        list_array.append(array[idx])
    return list_array


def crop_face_batch(imgs, batch_size):
    data_len = imgs.shape[0]
    hasFace = np.ones((data_len), dtype=bool)
    faces = np.ndarray((data_len, 3, 224, 224), dtype=np.float32)

    imgs = numpy_convert_to_list(imgs)

    batch_locs = fr.batch_face_locations(imgs, batch_size=batch_size)

    for idx, locations in enumerate(batch_locs):
        if len(locations) == 0:
            hasFace[idx] = False
            continue
        flags = locations[max_area_indx(locations)]
        cropped = cv.resize(imgs[idx][flags[0]:flags[2], flags[3]:flags[1]], (224, 224)).astype(np.float32)
        faces[idx] = np.transpose(cropped, (2, 0, 1))

    return faces, hasFace
