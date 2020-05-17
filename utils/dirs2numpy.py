import os
import argparse

import numpy as np
try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv


def crop_to_16vs9(imgs):
    if len(imgs.shape) == 3:
        imgs = imgs[np.newaxis, :, :, :]

    height = imgs.shape[1]
    width = imgs.shape[2]
    suppose_width = height * 16 // 9
    suppose_height = width * 9 // 16

    if width > suppose_width + 1:  # width is too long
        w_start = (width - suppose_width) // 2
        res = imgs[:, :, w_start:-w_start, :]
    elif width < suppose_width - 1:  # height is too long
        h_start = (height - suppose_height) // 2
        res = imgs[:, h_start:-h_start, :, :]
    else:
        res = imgs
    

    return res.squeeze()


def main(args):
    for idx in range(len(args.dirnames)):
        dname = os.path.join(args.dirpath, args.dirnames[idx])
        flist = os.listdir(dname)

        datalen = len(flist)
        imgs = np.ndarray((datalen, args.shape[0], args.shape[1], 3), dtype=np.uint8)
        for fidx in range(datalen):
            img = cv.imread(os.path.join(dname, flist[fidx]))
            if img is None:
                raise ValueError('Could not open file: {}'.format(os.path.join(dname, flist[fidx])))
            img = crop_to_16vs9(img)
            imgs[fidx] = cv.resize(img, (args.shape[1], args.shape[0]))

        np.save(os.path.join(args.savedir, args.dirnames[idx] + '.npy'), imgs)
        print('Save ' + args.dirnames[idx] + '.npy')
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='dirpath', default='/media/li-zhihang/LiZhihang/摄影标签细分/景别')
    parser.add_argument('-l', dest='dirnames', type=list, default=['远景', '全景', '中景', '中近景', '近景', '特写'])
    parser.add_argument('-s', dest='savedir', default='/media/li-zhihang/LiZhihang/摄影标签细分')
    parser.add_argument('--shape', type=tuple, default=(900, 1600))
    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        raise FileNotFoundError

    os.makedirs(args.savedir, exist_ok=True)

    main(args)
