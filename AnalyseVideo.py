import os

import cv2.cv2 as cv
import numpy as np
import tensorflow as tf

from methods.color.color import get_dominent_colors
from methods.lightpose.pose import LightPose
from methods.opt import opt
from methods.scale.scale import PoseRecog

args = opt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Analyser(object):
    def __init__(self, batch_size, cluster_num=5):
        self.cluster_num = cluster_num
        self.batch_size = batch_size
        self.light_reader = LightPose(self.batch_size)
        self.pose_reader = PoseRecog()
        print('Analyser built.')

    def get_shot_scale(self, imgs):
        self.pose_reader.get_pose(imgs)

    def get_light_pose(self, imgs):
        return self.light_reader.get_lightpose(imgs)

    def get_colors(self, imgs):
        data_len = imgs.shape[0]
        color_plate = np.ndarray((data_len, self.cluster_num, 3), dtype=int)
        for k in range(data_len):
            color_plate[k] = get_dominent_colors(imgs[k], self.cluster_num)
        return color_plate


def read_block(cap, win_len, sample_interval=1):
    width = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    height = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    imgs = np.ndarray((win_len, width, height, 3), dtype=np.uint8)
    for k in range(win_len):
        _, img = cap.read()
        if img is None:
            sample_index = [x for x in range(0, k, sample_interval)]
            return imgs[sample_index], True
        imgs[k] = img

    sample_index = [x for x in range(0, win_len, sample_interval)]
    return imgs[sample_index], False


def main():
    cap = cv.VideoCapture(args.video_path)
    fps = cap.get(cv.CAP_PROP_FPS)

    print('Video Name: {:s}\nFPS: {:.2f}\nSample Interval: {:d}\nSample Rate: {:.2f}\nWindow Length: {:d}\nBatch Size: {:d}'.format(
        args.video_path, fps, args.sample_interval, fps / args.sample_interval, args.win_len, args.batch_size))

    analyser = Analyser(args.batch_size)

    print('Start reading video...')
    eof_flag = False
    while not eof_flag:
        imgs, eof_flag = read_block(cap, args.win_len, args.sample_interval)
        analyser.get_colors(imgs)
        print('got color')
        analyser.get_light_pose(imgs)
        print('got light pose')
        analyser.get_shot_scale(imgs)
        print('got shot scale')

    cap.release()


if __name__ == '__main__':

    if not os.path.exists(args.video_path):
        raise FileNotFoundError('Cannot find the video.')

    main()
