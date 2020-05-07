import os

import cv2.cv2 as cv
import numpy as np
import tensorflow as tf

from methods.color.color import get_dominent_colors
from methods.lightpose.pose import LightPose
from methods.opt import opt
from methods.sat.sat import Sat_SVM
from methods.scale.scale import PoseRecog
from methods.symmetry.sym import get_sym_score
from methods.tone.tone import ToneClassifier

args = opt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Analyser(object):
    def __init__(self, batch_size, cluster_num=5):
        self.cluster_num = cluster_num
        self.batch_size = batch_size
        self.light_reader = LightPose(batch_size)
        self.pose_reader = PoseRecog()
        self.tone_reader = ToneClassifier(batch_size)
        self.sat_reader = Sat_SVM()
        print('Analyser built.')

    def get_shot_scale(self, imgs):
        return self.pose_reader.get_pose(imgs)

    def get_light_pose(self, imgs):
        return self.light_reader.get_lightpose(imgs)

    def get_colors(self, imgs):
        data_len = imgs.shape[0]
        color_plate = np.ndarray((data_len, self.cluster_num, 3), dtype=int)
        for k in range(data_len):
            color_plate[k] = get_dominent_colors(imgs[k], self.cluster_num)
        return color_plate

    def get_tone(self, imgs):
        return self.tone_reader.get_tone_class(imgs)

    def get_saturation(self, imgs):
        return self.sat_reader.get_sat(imgs)

    def get_symmetry(self, imgs):
        datalen = imgs.shape[0]
        sym = np.ndarray((datalen, 4))
        for k in range(datalen):
            sym[k] = get_sym_score(imgs[k])
        return sym


def read_block(cap, win_len, sample_interval=1):
    imgs = np.ndarray((win_len, args.processing_shape[0], args.processing_shape[1], 3), dtype=np.uint8)
    for k in range(win_len):
        _, img = cap.read()
        if img is None:
            sample_index = [x for x in range(0, k, sample_interval)]
            return imgs[sample_index], True
        imgs[k] = cv.resize(img, (args.processing_shape[1], args.processing_shape[0]))

    sample_index = [x for x in range(0, win_len, sample_interval)]
    return imgs[sample_index], False


def process(analyser, video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print('Cannot open source file: ' + video_path)
        return
    fps = cap.get(cv.CAP_PROP_FPS)

    print('*****************************************************')
    print('Video Name: {:s}\nFPS: {:.2f}\nSample Interval: {:d}\nSample Rate: {:.2f}\nWindow Length: {:d}\nBatch Size: {:d}'.format(
        args.video_path, fps, args.sample_interval, fps / args.sample_interval, args.win_len, args.batch_size))
    print('*****************************************************')

    dirname = os.path.dirname(video_path)
    basename = os.path.basename(video_path).split('.')[0]
    fcolor = open(os.path.join(dirname, basename + '.color'), 'w')

    print('Start reading video...')
    eof_flag = False
    while not eof_flag:
        imgs, eof_flag = read_block(cap, args.win_len, args.sample_interval)
        color = analyser.get_colors(imgs)
        print('got color')
        analyser.get_light_pose(imgs)
        print('got light pose')
        analyser.get_shot_scale(imgs)
        print('got shot scale')
        tone = analyser.get_tone(imgs)
        print('got tone type')
        sat = analyser.get_saturation(imgs)
        print('got sat type')
        sym = analyser.get_symmetry(imgs)
        print('got sym score')

        for k in range(imgs.shape[0]):
            for cluster in range(5):
                fcolor.write('{:3d} {:3d} {:3d}\n'.format(color[k, cluster, 0], color[k, cluster, 1], color[k, cluster, 2]))
            fcolor.write('{:d}\n'.format(tone[k]))
            fcolor.write('{:d}\n'.format(sat[k]))
            fcolor.write('{:.2f} {:.2f} {:.2f} {:.2f}\n'.format(sym[k, 0], sym[k, 1], sym[k, 2], sym[k, 3]))

    cap.release()
    fcolor.close()


def main():
    analyser = Analyser(args.batch_size)

    if os.path.isfile(args.video_path):
        process(analyser, args.video_path)
    else:
        for file in os.listdir(args.video_path):
            process(analyser, os.path.join(args.video_path, file))
    print('Done.')


if __name__ == '__main__':

    if not os.path.exists(args.video_path):
        raise FileNotFoundError('Cannot find the video.')

    main()
