try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv

import numpy as np


def show_plate(img, name, cluster_num, color_list):
    square_len = 50

    def form_square(color):
        return np.tile(color[np.newaxis, np.newaxis, :], (square_len, square_len, 1))

    color_plate = np.zeros((square_len * cluster_num, square_len, 3), dtype=np.uint8)
    for ci in range(cluster_num):
        color_plate[square_len * ci: square_len * (ci + 1)] = form_square(color_list[ci])

    print('RGB of center colors:')
    print(color_list.shape)
    for ci in range(cluster_num):
        print(color_list[ci, 2], color_list[ci, 1], color_list[ci, 0], sep=', ')

    cv.imwrite(name + '.png', color_plate)


def get_dominent_colors(img, cluster_num=5, cluster='kmeans', if_show=False, name=None):
    pixels = img.reshape(-1, 3).astype(np.float32)

    if cluster == 'kmeans':
        criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        flags = cv.KMEANS_RANDOM_CENTERS
        _, _, centers = cv.kmeans(pixels, cluster_num, None, criteria, 1, flags)
    elif cluster == 'fcm':
        import skfuzzy as skf
        pixels = np.transpose(pixels, (1, 0))
        cmeans_res = skf.cmeans(pixels, cluster_num, 2, 1e-4, 100)
        centers = cmeans_res[0]
    else:
        raise NotImplementedError('Unrecognised cluster method.')

    color_list = np.zeros((cluster_num, 3), dtype='int')
    for c in range(cluster_num):
        color_list[c] = centers[c, ::-1].astype('int')

    if if_show:
        show_plate(img, name, cluster_num, color_list)

    return color_list
