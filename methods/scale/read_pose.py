import argparse
import json
import os

import keras
import numpy as np

from ..opt import opt

try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv


cn_name = ['case 2', 'case 1', '没这个选项', '大远景', '远景', '全景', '中景', '中近景', '近景', '特写']


def single_person(kp, height, logout=True):
    threshold1 = 0.25

    Nose = kp[0:3]
    LEye = kp[3:6]
    REye = kp[6:9]
    LEar = kp[9:12]
    REar = kp[12:15]
    LShoulder = kp[15:18]
    RShoulder = kp[18:21]
    LElbow = kp[21:24]
    RElbow = kp[24:27]
    LWrist = kp[27:30]
    RWrist = kp[30:33]
    LHip = kp[33:36]
    RHip = kp[36:39]
    LKnee = kp[39:42]
    RKnee = kp[42:45]
    LAnkle = kp[45:48]
    RAnkle = kp[48:51]

    # max score of body parts
    ankel = max(RAnkle[2], LAnkle[2])
    knee = max(RKnee[2], LKnee[2])
    hip = max(RHip[2], LHip[2])
    wrist = max(RWrist[2], LWrist[2])
    elbow = max(RElbow[2], LElbow[2])
    shoulder = max(RShoulder[2], LShoulder[2])
    ear = max(REar[2], LEar[2])
    eye = max(REye[2], LEye[2])
    nose = Nose[2]

    # compute the height of person, pickout far shots
    if eye > threshold1:
        upper = LEye[1] if LEye[2] > REye[2] else REye[1]
    elif nose > threshold1:
        upper = Nose[1]
    elif ear > threshold1:
        upper = LEar[1] if LEar[2] > REar[2] else REar[1]
    else:
        upper = None  # no need to find if undetectable

    if ankel > threshold1:
        lower = LAnkle[1] if LAnkle[2] > RAnkle[2] else RAnkle[1]
    elif knee > threshold1:
        lower = LKnee[1] if LKnee[2] > RKnee[2] else RKnee[1]
    elif hip > threshold1:
        lower = LHip[1] if LHip[2] > RHip[2] else RHip[1]
    else:
        lower = None  # no need to find if undetectable

    if upper is not None and lower is not None:
        hp = lower - upper
        h_ratio = hp / height
        if logout:
            print('height of person:{:.0f}, image height:{}, height ratio:{:.2f}'.format(hp, height, h_ratio))
    else:
        h_ratio = None

    if h_ratio is not None:
        if h_ratio < 0.3:
            t = 0
        elif h_ratio < 0.6:
            t = 1
        else:
            if ankel > threshold1:
                t = 2
            elif knee > threshold1:
                t = 3
            elif hip > threshold1:
                t = 4
            else:
                t = -2  # case 1
    else:
        if shoulder > threshold1:
            if elbow > threshold1:
                t = 4
            else:
                t = 5
        elif eye > threshold1 or ear > threshold1 or nose > threshold1:
            t = 6
        else:
            t = -3  # case 2

    if logout:
        print('ankel:{:.2f}, knee:{:.2f}, hip:{:.2f}, wrist:{:.2f}, elbow:{:.2f}, shoulder:{:.2f}, ear:{:.2f}, eye:{:.2f}, nose:{:.2f}'.format(
              ankel, knee, hip, wrist, elbow, shoulder, ear, eye, nose))
        print(cn_name[t + 3])

    return t


def read_json(path, img_dir):
    with open(path, 'r') as f:
        res = json.load(f)
    if res is None:
        print('\033[31m[FATAL] None recognizable person in current picture.\033[37m')
    else:
        print('Reading image "' + res[0]['image_id'])
        t = cv.imread(os.path.join(img_dir, res[0]['image_id']))
        height = t.shape[0]
        for per in res:
            if per['score'] < 1.5:
                print('\033[33mperson score {:.2f} below threshold\033[37m'.format(per['score']))
            else:
                print('person score: {:.2f}'.format(per['score']))
                single_person(per['keypoints'], height)


def read_pose(all_res, height):
    data_len = len(all_res)
    scale_levs = np.ndarray((data_len,), dtype=int)
    for im_idx in range(data_len):
        im_res = all_res[im_idx]['result']

        if im_res == [] or im_res is None:  # no human
            scale_levs[im_idx] = -1
            continue

        hu_num = len(im_res)
        hu_scales = np.ndarray((hu_num,), dtype=int)
        for hu_idx in range(hu_num):
            pro_scores = float(im_res[hu_idx]['proposal_score'])
            if pro_scores < 1.5:  # invalid human
                hu_scales[hu_idx] = -1
                continue

            kp_preds = im_res[hu_idx]['keypoints']
            kp_scores = im_res[hu_idx]['kp_score']
            keypoints = []
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            hu_scales[hu_idx] = single_person(keypoints, height, logout=False)

        scale_levs[im_idx] = np.max(hu_scales)
    return scale_levs


def perceptron():
    net = keras.models.Sequential([
        keras.layers.Dense(200, input_shape=(52,), activation='relu'),
        keras.layers.Dense(6, activation='softmax')
    ])
    return net


class PoseReader(object):
    def __init__(self):
        self.model = perceptron()
        self.model.load_weights('./models/scale_mlp/model_09.hdf5')

    def read_pose_mlp(self, final_result, height):
        datalen = len(final_result)
        pose_raw_list = np.ndarray((0, 52))

        scale_type = np.ndarray((datalen,), dtype=int)
        human_num = np.zeros((datalen,), dtype=int)
        wait_decide = np.ones((datalen,), dtype=bool)
        pose_raw_list = np.zeros((datalen, 52))
        for im_idx in range(datalen):
            im_res = final_result[im_idx]['result']
            if im_res == [] or im_res is None:  # no human 1
                scale_type[im_idx] = -1
                wait_decide[im_idx] = False
                continue

            hu_num = len(im_res)
            if hu_num == 0:
                scale_type[im_idx] = -1
                wait_decide[im_idx] = False
                continue

            hum_scores = np.zeros((hu_num,))
            for hu_idx in range(hu_num):
                hum_scores[hu_idx] = float(im_res[hu_idx]['proposal_score'])

            if np.max(hum_scores) < 0.5:  # no human
                scale_type[im_idx] = -2
                wait_decide[im_idx] = False
                continue

            human_num[im_idx] = np.sum(hum_scores >= 0.5)

            hu_max_idx = np.argmax(hum_scores)
            kp_preds = im_res[hu_max_idx]['keypoints']
            kp_scores = im_res[hu_max_idx]['kp_score']
            keypoints = []
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            keypoints.append(height)

            pose_raw_list[im_idx] = np.array(keypoints)

        logits = np.ndarray((datalen,), dtype=int)
        pred = self.model.predict(pose_raw_list, batch_size=opt.batch_size)
        logits = np.argmax(pred, axis=-1)
        scale_type[wait_decide] = logits[wait_decide]
        return scale_type, human_num


def main(args):
    for file in os.listdir(args.json_path):
        filepath = os.path.join(args.json_path, file)
        if os.path.isfile(filepath):
            read_json(filepath, args.img_path)
            print('---------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_path', type=str, default='./AlphaPose/res/')
    parser.add_argument('--img_path', type=str, default='./AlphaPose/res/vis/')

    args = parser.parse_args()

    main(args)
