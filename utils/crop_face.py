import argparse
import os

import face_recognition as fr
import numpy as np

try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv


pos_list = ['侧光', '侧逆光', '前侧光', '脚光', '顶光', '顺光']
pos_list_en = ['sl', 'bsl', 'fsl', 'fol', 'tl', 'frl']
# above abbrs stand for: sidelight, back-sidelight, front-sidelight,
# footlight, backlight, toplight, frontlight


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


def ReadFolder(read_path, train_dir, test_dir, pos, vis=False, vis_dir=None):
    '''read position file. crop faces and save them.'''

    files = os.listdir(read_path)
    test_files = files[-len(files) // 8:]
    train_files = files[:-len(files) // 8]

    # build training set
    index = 0
    faces = np.zeros((len(train_files) * 6, 224, 224, 3), dtype=int)
    for file in train_files:
        image = cv.imread(os.path.join(read_path, file))  # [h, w, c]
        # find faces
        try:
            locations = fr.api.face_locations(image, model='cnn')
        except Exception as e:
            print('\033[31m[FATAL] Locations in file {} cannot be extracted due to folloing reason:\n{}\033[37m'.format(os.path.join(read_path, file), e))
        else:
            if not locations:
                print('\033[33m[WARN] Cannot find face in file {} \033[37m'.format(os.path.join(read_path, file)))
                continue
            flags = locations[max_area_indx(locations)]

            cropped = cv.resize(image[flags[0]:flags[2], flags[3]:flags[1], :], (224, 224))
            h_flip = cv.flip(cropped, 1)  # horizontal flip
            m = cv.getRotationMatrix2D((112, 112), 10, 1)
            rotate1 = cv.warpAffine(cropped, m, (224, 224))
            m = cv.getRotationMatrix2D((112, 112), -10, 1)
            rotate2 = cv.warpAffine(cropped, m, (224, 224))
            zoomin = cv.resize(cropped[25:-25, 25:-25], (224, 224))
            noise = (np.random.randn(224, 224, 3) * 20 + cropped).clip(0, 255)

            if vis:
                cv.imwrite(os.path.join(vis_dir, 'train_' + str(index) + '.png'), cropped)

            faces[index] = cropped
            index += 1
            faces[index] = h_flip
            index += 1
            faces[index] = rotate1
            index += 1
            faces[index] = rotate2
            index += 1
            faces[index] = zoomin
            index += 1
            faces[index] = noise
            index += 1

    np.save(os.path.join(train_dir, pos), faces)
    print('[INFO] ' + os.path.join(train_dir, pos) + ' contains {} images'.format(faces.shape[0]))

    # build testing set
    index = 0
    faces = np.zeros((len(test_files), 224, 224, 3), dtype=int)
    for file in test_files:
        image = cv.imread(os.path.join(read_path, file))  # [h, w, c]
        # find faces
        try:
            locations = fr.api.face_locations(image, model='cnn')
        except Exception as e:
            print('\033[31m[FATAL] Locations in file {} cannot be extracted due to folloing reason:\n{}\033[37m'.format(os.path.join(read_path, file), e))
        else:
            if not locations:
                print('\033[33m[WARN] Cannot find face in file {} \033[37m'.format(os.path.join(read_path, file)))
                continue
            flags = locations[max_area_indx(locations)]

            cropped = cv.resize(image[flags[0]:flags[2], flags[3]:flags[1], :], (224, 224))

            if vis:
                cv.imwrite(os.path.join(vis_dir, 'test_' + str(index) + '.png'), cropped)

            faces[index] = cropped
            index += 1

    np.save(os.path.join(test_dir, pos), faces)
    print('[INFO] ' + os.path.join(test_dir, pos) + ' contains {} images'.format(faces.shape[0]))


def main(args):
    for pos_idx in range(len(pos_list)):
        pos = pos_list[pos_idx]
        pos_en = pos_list_en[pos_idx]
        read_path = os.path.join(args.root, pos)
        ReadFolder(read_path, args.train_dir, args.test_dir, pos_en, args.vis, os.path.join(args.vis_dir, pos_en))
        print('[INFO] Finish reading {}'.format(read_path))
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../Data/light-pos')
    parser.add_argument('--save_dir', type=str, default='./dataset/light-pos')
    parser.add_argument('--vis', type=bool, default=True, help='visualize')
    args = parser.parse_args()

    args.train_dir = os.path.join(args.save_dir, 'train')
    args.test_dir = os.path.join(args.save_dir, 'test')
    args.vis_dir = os.path.join(args.save_dir, 'vis')

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    if args.vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        for lp in pos_list_en:
            os.makedirs(os.path.join(args.vis_dir, lp), exist_ok=True)

    main(args)
