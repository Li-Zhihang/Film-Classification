import os
import argparse

import numpy as np
try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv


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
            imgs[fidx] = cv.resize(img, (args.shape[1], args.shape[0]))

        np.save(os.path.join(args.savedir, args.dirnames[idx] + '.npy'), imgs)
        print('Save ' + args.dirnames[idx] + '.npy')
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='dirpath', default='/media/li-zhihang/LiZhihang/摄影标签细分/景别')
    parser.add_argument('-l', dest='dirnames', type=list, default=['大远景', '远景', '全景', '中景', '中近景', '近景', '特写'])
    parser.add_argument('-s', dest='savedir', default='/media/li-zhihang/LiZhihang/摄影标签细分')
    parser.add_argument('--shape', type=tuple, default=(360, 640))
    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        raise FileNotFoundError

    os.makedirs(args.savedir, exist_ok=True)

    main(args)
