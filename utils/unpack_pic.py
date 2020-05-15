import argparse
import os

import numpy as np

try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv


def read_np(file, save_dir):
    img = np.load(file)
    name = os.path.basename(file)
    print('read ' + file)
    for idx in range(img.shape[0]):
        cv.imwrite(os.path.join(save_dir, name.split('.')[0] + '_' + str(idx) + '.png'), img[idx])


def main(args):
    if os.path.isdir(args.tgt):
        for f in os.listdir(args.tgt):
            if f.split('.')[-1] == 'npy':
                read_np(os.path.join(args.tgt, f), args.save_dir)
            else:
                print(os.path.join(args.tgt, f) + ' is not a numpy file.')
        print('Done.')
    else:
        f = args.tgt
        if f.split('.')[-1] == 'npy':
            read_np(f, args.save_dir)
        else:
            print(f + ' is not a numpy file.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt', type=str, default='./dataset/light/packed')
    parser.add_argument('--save_dir', type=str, default='./dataset/light/vis')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
