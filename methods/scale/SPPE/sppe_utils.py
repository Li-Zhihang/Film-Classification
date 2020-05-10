try:
    import cv2.cv2 as cv
except Exception:
    import cv2 as cv

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from ...opt import opt


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def im_to_torch(img):
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def cropBox(img, ul, br, resH, resW):
    ul = ul.int()
    br = (br - 1).int()
    # br = br.int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    box_shape = [(br[1] - ul[1]).item(), (br[0] - ul[0]).item()]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    if ul[1] > 0:
        img[:, :ul[1], :] = 0
    if ul[0] > 0:
        img[:, :, :ul[0]] = 0
    if br[1] < img.shape[1] - 1:
        img[:, br[1] + 1:, :] = 0
    if br[0] < img.shape[2] - 1:
        img[:, :, br[0] + 1:] = 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array(
        [ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array(
        [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv.warpAffine(torch_to_im(img), trans, (resW, resH), flags=cv.INTER_LINEAR)

    return im_to_torch(torch.Tensor(dst_img))


def transformBoxInvert(pt, ul, br, inpH, inpW, resH, resW):
    center = np.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = (pt * lenH) / resH
    _pt[0] = _pt[0] - max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] - max(0, (lenH - 1) / 2 - center[1])

    new_point = np.zeros(2)
    new_point[0] = _pt[0] + ul[0]
    new_point[1] = _pt[1] + ul[1]
    return new_point


def findPeak(hm):
    mx = maximum_filter(hm, size=5)
    idx = zip(*np.where((mx == hm) * (hm > 0.1)))
    candidate_points = []
    for (y, x) in idx:
        candidate_points.append([x, y, hm[y][x]])
    if len(candidate_points) == 0:
        return torch.zeros(0)
    candidate_points = np.array(candidate_points)
    candidate_points = candidate_points[np.lexsort(-candidate_points.T)]
    return torch.Tensor(candidate_points)


def processPeaks(candidate_points, hm, pt1, pt2, inpH, inpW, resH, resW):
    # (Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> List[Tensor]

    if candidate_points.shape[0] == 0:  # Low Response
        maxval = np.max(hm.reshape(1, -1), 1)
        idx = np.argmax(hm.reshape(1, -1), 1)

        x = idx % resW
        y = int(idx / resW)

        candidate_points = np.zeros((1, 3))
        candidate_points[0, 0:1] = x
        candidate_points[0, 1:2] = y
        candidate_points[0, 2:3] = maxval

    res_pts = []
    for i in range(candidate_points.shape[0]):
        x, y, maxval = candidate_points[i][0], candidate_points[i][1], candidate_points[i][2]

        if bool(maxval < 0.05) and len(res_pts) > 0:
            pass
        else:
            if bool(x > 0) and bool(x < resW - 2):
                if bool(hm[int(y)][int(x) + 1] - hm[int(y)][int(x) - 1] > 0):
                    x += 0.25
                elif bool(hm[int(y)][int(x) + 1] - hm[int(y)][int(x) - 1] < 0):
                    x -= 0.25
            if bool(y > 0) and bool(y < resH - 2):
                if bool(hm[int(y) + 1][int(x)] - hm[int(y) - 1][int(x)] > 0):
                    y += (0.25 * inpH / inpW)
                elif bool(hm[int(y) + 1][int(x)] - hm[int(y) - 1][int(x)] < 0):
                    y -= (0.25 * inpH / inpW)

            # pt = torch.zeros(2)
            pt = np.zeros(2)
            pt[0] = x + 0.2
            pt[1] = y + 0.2

            pt = transformBoxInvert(pt, pt1, pt2, inpH, inpW, resH, resW)

            res_pt = np.zeros(3)
            res_pt[:2] = pt
            res_pt[2] = maxval

            res_pts.append(res_pt)

            if maxval < 0.05:
                break
    return res_pts


def getMultiPeakPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):

    assert hms.dim() == 4, 'Score maps should be 4-dim'

    preds_img = {}
    hms = hms.numpy()
    for n in range(hms.shape[0]):        # Number of samples
        preds_img[n] = {}           # Result of sample: n
        for k in range(hms.shape[1]):    # Number of keypoints
            preds_img[n][k] = []    # Result of keypoint: k
            hm = hms[n][k]

            candidate_points = findPeak(hm)

            res_pt = processPeaks(candidate_points, hm,
                                  pt1[n], pt2[n], inpH, inpW, resH, resW)

            preds_img[n][k] = res_pt

    return preds_img


def transformBoxInvert_batch(pt, ul, br, inpH, inpW, resH, resW):
    '''
    pt:     [n, 17, 2]
    ul:     [n, 2]
    br:     [n, 2]
    '''
    center = (br - 1 - ul) / 2

    size = br - ul
    size[:, 0] *= (inpH / inpW)

    lenH, _ = torch.max(size, dim=1)   # [n,]
    lenW = lenH * (inpW / inpH)

    _pt = (pt * lenH[:, np.newaxis, np.newaxis]) / resH
    _pt[:, :, 0] = _pt[:, :, 0] - ((lenW[:, np.newaxis].repeat(1, 17) - 1) /
                                   2 - center[:, 0].unsqueeze(-1).repeat(1, 17)).clamp(min=0)
    _pt[:, :, 1] = _pt[:, :, 1] - ((lenH[:, np.newaxis].repeat(1, 17) - 1) /
                                   2 - center[:, 1].unsqueeze(-1).repeat(1, 17)).clamp(min=0)

    new_point = torch.zeros(pt.size())
    new_point[:, :, 0] = _pt[:, :, 0] + ul[:, 0].unsqueeze(-1).repeat(1, 17)
    new_point[:, :, 1] = _pt[:, :, 1] + ul[:, 1].unsqueeze(-1).repeat(1, 17)
    return new_point


def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    '''
    Get keypoint location from heatmaps
    '''

    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(round(float(preds[i][j][1])))
            if 0 < pX < opt.outputResW - 1 and 0 < pY < opt.outputResH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                preds[i][j] += diff.sign() * 0.25
    preds += 0.2

    preds_tf = torch.zeros(preds.size())

    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW, resH, resW)

    return preds, preds_tf, maxval
