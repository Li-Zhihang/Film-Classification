import torch
import numpy as np


def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    # Softmax the class scores
    prediction[:, :, 5: 5 +
               num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction


def dynamic_write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    prediction_bak = prediction.clone()
    dets = write_results(prediction.clone(), confidence,
                         num_classes, nms, nms_conf)
    if isinstance(dets, int):
        return dets

    if dets.shape[0] > 100:
        nms_conf -= 0.05
        dets = write_results(prediction_bak.clone(),
                             confidence, num_classes, nms, nms_conf)

    return dets


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area

    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * \
        torch.max(inter_rect_y2 - inter_rect_y1 + 1,
                  torch.zeros(inter_rect_x2.shape).cuda())

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().float().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except Exception:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    write = False
    num = 0
    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]

        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(
            image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Get rid of the zero entries
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        # Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:, -1])
        except Exception:
            continue

        # WE will do NMS classwise
        # print(img_classes)
        for cls in img_classes:
            if cls != 0:
                continue
            # get the detections with one particular class
            cls_mask = image_pred_ * \
                (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]

            # if nms has to be done
            if nms:
                # Perform non-maximum suppression
                max_detections = []
                while image_pred_class.size(0):
                    # Get detection with highest confidence and save as max detection
                    max_detections.append(image_pred_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(image_pred_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_iou(max_detections[-1], image_pred_class[1:])
                    # Remove detections with IoU >= NMS threshold
                    image_pred_class = image_pred_class[1:][ious < nms_conf]

                image_pred_class = torch.cat(max_detections).data

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear straucture to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column

            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
            num += 1

    if not num:
        return 0

    return output
