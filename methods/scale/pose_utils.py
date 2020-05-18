import math
import os
import time
from queue import Queue
from threading import Thread

import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn

from ..opt import opt
from .matching import candidate_reselect as matching
from .pPose_nms import pose_nms
from .SPPE.fastpose import FastPose
from .SPPE.sppe_utils import (cropBox, getMultiPeakPrediction, getPrediction,
                              im_to_torch)
from .yolo.dartnet import Darknet
from .yolo.yolo_utils import dynamic_write_results

try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    new_w = int(img_w * min(inp_dim / img_w, inp_dim / img_h))
    new_h = int(img_h * min(inp_dim / img_w, inp_dim / img_h))
    resized_image = cv.resize(
        img, (new_w, new_h), interpolation=cv.INTER_CUBIC)

    canvas = np.full((inp_dim, inp_dim, 3), 128)
    canvas[(inp_dim - new_h) // 2:(inp_dim - new_h) // 2 + new_h,
           (inp_dim - new_w) // 2:(inp_dim - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(orig_im, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(orig_im, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0)
    return img_


class ImageLoader:
    def __init__(self, batchSize=1, queueSize=50):

        self.batchSize = batchSize
        # initialize the queue used to store data
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def reset(self):
        self.Q.queue.clear()
        self.imgs = None
        self.dim = None
        self.datalen = None
        self.num_batches = None

    def feed(self, imgs, dim):
        self.orig_imgs = imgs
        self.dim = dim
        self.datalen = imgs.shape[0]
        leftover = 0
        if (self.datalen) % self.batchSize:
            leftover = 1
        self.num_batches = self.datalen // self.batchSize + leftover

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            p = Thread(target=self._getitem_yolo, args=())
        else:
            p = mp.Process(target=self._getitem_yolo, args=())

        p.daemon = True
        p.start()
        return self

    def _getitem_yolo(self):
        for i in range(self.num_batches):
            start_batch = i * self.batchSize
            end_batch = min((i + 1) * self.batchSize, self.datalen)
            img = torch.zeros((end_batch - start_batch, 3, opt.inp_dim, opt.inp_dim))
            for k in range(end_batch - start_batch):
                img_k = prep_image(self.orig_imgs[k], opt.inp_dim)
                img[k] = img_k

            while self.Q.full():
                time.sleep(2)

            orig = self.orig_imgs[start_batch: end_batch]
            self.Q.put((img, orig))

    def getitem(self):
        return self.Q.get()

    def length(self):
        return self.datalen

    def len(self):
        return self.Q.qsize()

    def getdim(self):
        return self.dim


class InferenNet_fast(nn.Module):
    def __init__(self):
        super(InferenNet_fast, self).__init__()

        model = FastPose()
        model.load_state_dict(torch.load('./models/SPPE/duc_se.pth'))
        print('successfully load inferenNet.')
        self.pyranet = model

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        return out


class DetectionLoader:
    def __init__(self, dataloader, batchSize=1, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("./models/yolo/yolov3-spp.cfg")
        self.det_model.load_weights('./models/yolo/yolov3-spp.weights')
        print('successfully load detection model.')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = opt.inp_dim
        self.det_model.cuda()
        self.det_model.eval()

        self.dataloader = dataloader
        self.batchSize = batchSize
        # initialize the queue used to store frames read from the video file
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def init(self):
        self.stopped = False
        self.datalen = self.dataloader.length()
        self.dim = self.dataloader.getdim()
        leftover = 0
        if (self.datalen) % self.batchSize:
            leftover = 1
        self.num_batches = self.datalen // self.batchSize + leftover

        self.Q.queue.clear()

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.num_batches):
            img, orig = self.dataloader.getitem()
            im_dim_list = torch.Tensor(self.dim).repeat(img.shape[0], 2)
            if img is None:
                self.Q.put((None, None, None, None, None, None))
                return

            with torch.no_grad():
                # Human Detection
                img = img.cuda()
                prediction = self.det_model(img, CUDA=True)
                # NMS process
                dets = dynamic_write_results(
                    prediction, opt.confidence, opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(orig.shape[0]):
                        if self.Q.full():
                            time.sleep(2)
                        self.Q.put((orig[k], None, None, None, None, None))
                    continue
                dets = dets.cpu()
                im_dim_list = torch.index_select(
                    im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(
                    self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor *
                                    im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor *
                                    im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(
                        dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(
                        dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

            for k in range(orig.shape[0]):
                boxes_k = boxes[dets[:, 0] == k]
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    if self.Q.full():
                        time.sleep(2)
                    self.Q.put((orig[k], None, None, None, None, None))
                    continue
                inps = torch.zeros(boxes_k.size(
                    0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes_k.size(0), 2)
                pt2 = torch.zeros(boxes_k.size(0), 2)
                if self.Q.full():
                    time.sleep(2)
                self.Q.put(
                    (orig[k], boxes_k, scores[dets[:, 0] == k], inps, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft,
                              bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2


class DetectionProcessor:
    def __init__(self, detectionLoader, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.detectionLoader = detectionLoader
        # initialize the queue used to store data
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)

    def init(self):
        self.Q.queue.clear()
        self.stopped = False
        self.datalen = self.detectionLoader.datalen

    def start(self):
        # start a thread to read frames from the file video stream
        if opt.sp:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        else:
            p = mp.Process(target=self.update, args=())
            p.daemon = True
            p.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.datalen):

            with torch.no_grad():
                (orig_img, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()
                if orig_img is None:
                    self.Q.put((None, None, None, None, None, None))
                    return
                if boxes is None or boxes.nelement() == 0:
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((None, orig_img, boxes, scores, None, None))
                    continue
                inp = im_to_torch(cv.cvtColor(orig_img, cv.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put((inps, orig_img, boxes, scores, pt1, pt2))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)


def vis_frame(frame, im_res, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191,
                                                                                    255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135,
                                                       255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif format == 'mpii':
        l_pair = [
            (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
            (13, 14), (14, 15), (3, 4), (4, 5),
            (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
        ]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED,
                   RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE,
                      RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    img = frame
    height, width = img.shape[:2]
    img = cv.resize(img, (int(width / 2), int(height / 2)))
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x / 2), int(cor_y / 2))
            bg = img.copy()
            cv.circle(bg, (int(cor_x / 2), int(cor_y / 2)), 2, p_color[n], -1)
            # Now create a mask of logo and create its inverse mask also
            transparency = max(0, min(1, kp_scores[n]))
            img = cv.addWeighted(bg, transparency, img, 1 - transparency, 0)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv.ellipse2Poly((int(mX), int(mY)), (int(
                    length / 2), stickwidth), int(angle), 0, 360, 1)
                cv.fillConvexPoly(bg, polygon, line_color[i])
                # cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                transparency = max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])))
                img = cv.addWeighted(bg, transparency, img, 1 - transparency, 0)
    img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
    return img


class DataWriter(object):
    def __init__(self, queueSize=1024):
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

    def init(self):
        self.Q.queue.clear()
        self.stopped = False
        self.final_result = []
        self.write_count = 0
        self.T = time.strftime("%Y%m%d", time.localtime())

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                (boxes, scores, hm_data, pt1, pt2, orig_img) = self.Q.get()
                orig_img = np.array(orig_img, dtype=np.uint8)
                self.write_count += 1
                im_name = self.T + '_' + str(self.write_count) + '.jpg'
                if boxes is None:
                    result = {
                        'imgname': im_name,
                        'result': []
                    }
                    self.final_result.append(result)
                    if opt.save_img or opt.save_video or opt.vis:
                        img = orig_img
                        if opt.vis:
                            cv.imshow("AlphaPose Demo", img)
                            cv.waitKey(30)
                        if opt.save_img:
                            cv.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
                else:
                    # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                    if opt.matching:
                        preds = getMultiPeakPrediction(
                            hm_data, pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                        result = matching(boxes, scores.numpy(), preds)
                    else:
                        preds_hm, preds_img, preds_scores = getPrediction(
                            hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
                        result = pose_nms(boxes, scores, preds_img, preds_scores)
                    result = {
                        'imgname': im_name,
                        'result': result
                    }
                    self.final_result.append(result)
                    if opt.save_img or opt.save_video or opt.vis:
                        img = vis_frame(orig_img, result)
                        if opt.vis:
                            cv.imshow("AlphaPose Demo", img)
                            cv.waitKey(30)
                        if opt.save_img:
                            cv.imwrite(os.path.join(opt.outputpath, 'vis', im_name), img)
                        if opt.save_video:
                            self.stream.write(img)
            else:
                time.sleep(0.1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img):
        # save next frame in the queue
        self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()


def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval
