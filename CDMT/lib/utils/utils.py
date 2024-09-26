# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
import math
import numpy as np

import torch
import torch.nn as nn


def point2line(point1,point2):
    # 用极坐标 r,rho 来表示过两点的直线

    # 提取坐标值
    x1, y1 = point1
    x2, y2 = point2

    # 计算极径（r）原点到直线的距离
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    # 计算原点到直线的距离（点到直线的距离公式）
    r = np.abs(C) / np.sqrt(A**2 + B**2)

    # direction 切点位于y轴的上方还是下方，也就是直线的截距
    direction = (y1*x2-y2*x1)*(x2-x1)
    if direction >=0:
        direction = 1
    else:
        direction = -1

    r*= direction

    # 计算极角（theta）（以弧度为单位）
    theta = np.arctan2(y2 - y1, x2 - x1)+np.pi/2    # 切线和直线垂直
    # +pi/2是把极轴从x轴转为y轴

    # 将极角从弧度转换为度
    theta_degrees = np.degrees(theta)

    return r,theta_degrees  # 其实是极坐标表示的切点位置，从y轴顺时针旋转

def get_transform(center, scale, output_size, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(output_size[1]) / h
    t[1, 1] = float(output_size[0]) / h
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1]/2
        t_mat[1, 2] = -output_size[0]/2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1



def transform_preds(coords, center, scale, output_size):

    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.tensor(transform_pixel(coords[p, 0:2], center, scale, output_size, 1, 0))
    return coords

def decode_keypoint_complex(keypoint, meta):
    for p in range(keypoint.shape[0]):
        output_size = [meta['heatmap_size'][0][p], meta['heatmap_size'][1][p]]
        center = [meta['crop_locate'][0][p], meta['crop_locate'][1][p]]
        center[0] = center[0]+2*output_size[0]
        center[1] = center[1]+2*output_size[1]
        scale = meta['scale'][p]
        keypoint[p] = torch.tensor(transform_preds(keypoint[p], center, scale, output_size))
    return keypoint    

def decode_line(line, meta):
    line[:,:,1]*= 180
    line[:,:,0]*= meta['heatmap_size'][0][:,None]*4
    y = meta['crop_locate'][0][:,None]
    x = meta['crop_locate'][1][:,None]
    line[:,:,0] = line[:,:,0] + y*np.sin(line[:,:,1]/180*np.pi) + x*np.cos(line[:,:,1]/180*np.pi)
    line[:,:,0]/= (meta['scale'][0]+meta['scale'][1])[:,None]/2
    return line

def decode_keypoint(keypoint, meta):
    keypoint = keypoint*4
    keypoint[:,:,0]+= meta['crop_locate'][0][:,None]
    keypoint[:,:,1]+= meta['crop_locate'][1][:,None]
    keypoint[:,:,0]/=meta['scale'][0][:,None]
    keypoint[:,:,1]/=meta['scale'][1][:,None]
    return torch.round(keypoint)

def decode_heatmap(heatmap):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    def heatmap2coord(heatmap):
        scores = heatmap
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1

        pred_coord = idx.repeat(1, 1, 2).float()
        '''
        pred_coord[:, :, 0] = (pred_coord[:, :, 0] - 1) % scores.size(3) 
        pred_coord[:, :, 1] = torch.floor((pred_coord[:, :, 1] - 1) / scores.size(3)) 
        '''
        pred_coord[:, :, 0] = (pred_coord[:, :, 0] - 1) // scores.size(3)
        pred_coord[:, :, 1] = (pred_coord[:, :, 1] - 1) % scores.size(3) 
        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        pred_coord *= pred_mask
        return pred_coord

    def pose_process(heatmap, coords):
        # pose-processing
        '''
        for n in range(coords.size(0)):
            for p in range(coords.size(1)):
                hm = heatmap[n][p]
                px = int(math.floor(coords[n][p][1]))
                py = int(math.floor(coords[n][p][0]))
                if (px > 1) and (px < hm.shape[0]) and (py > 1) and (py < hm.shape[1]):
                    diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                    coords[n][p] += diff.sign() * .25
        '''
        coords += 0.5
        return coords

    coords = heatmap2coord(heatmap)
    coords = coords.cpu()
    preds = pose_process(heatmap, coords).clone().to(heatmap.device)
    return preds


class FullModel(nn.Module):
  """
  Distribute the loss on multi-gpu to reduce 
  the memory cost in the main gpu.
  You can check the following discussion.
  https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
  """
  def __init__(self, model, loss):
    super(FullModel, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, inputs, labels, *args, **kwargs):
    seg_output, keypoint_output = self.model(inputs, *args, **kwargs)   # (B,5,130,130)
    loss = self.loss(outputs, labels)       # label:(B,520,520)
    return torch.unsqueeze(loss,0), outputs

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            cfg_name / time_str
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr