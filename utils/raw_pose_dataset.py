import argparse
import os

import numpy as np
import tensorflow as tf

from methods.scale.scale import PoseRecog

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
scale_name_cn = ['远景', '全景', '中景', '中近景', '近景', '特写']


def main(args):
    pose_reader = PoseRecog()

    for idx in range(len(scale_name_cn)):
        imgs = np.load(os.path.join(args.data_dir, scale_name_cn[idx] + '.npy'))

        final_result, height = pose_reader.get_pose(imgs, True)
        pose_raw_list = np.ndarray((0, 52))
        datalen = len(final_result)
        for im_idx in range(datalen):
            im_res = final_result[im_idx]['result']
            if im_res == [] or im_res is None:
                continue
            hu_num = len(im_res)
            hum_scores = np.zeros((hu_num,))
            for hu_idx in range(hu_num):
                hum_scores[hu_idx] = float(im_res[hu_idx]['proposal_score'])
            hu_max = np.max(hum_scores)
            if hu_max < 1.5:
                continue

            hu_max_idx = np.argmax(hum_scores)
            kp_preds = im_res[hu_max_idx]['keypoints']
            kp_scores = im_res[hu_max_idx]['kp_score']
            keypoints = []
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            keypoints.append(height)

            keypoints = np.array(keypoints)
            keypoints = keypoints[np.newaxis, :]
            pose_raw_list = np.concatenate((pose_raw_list, keypoints), axis=0)
        np.save(os.path.join(args.output_dir, 'pose_' + scale_name_cn[idx] + '.npy'), pose_raw_list)
        print('save raw pose type {} of {:d} images'.format(scale_name_cn[idx], len(pose_raw_list)))
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='data_dir', default='H:\摄影标签细分')
    parser.add_argument('-o', dest='output_dir', default='D:\scale_dataset')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
