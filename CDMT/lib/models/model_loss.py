import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from core.criterion import OhemCrossEntropy, CrossEntropy

class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce 
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """
    def __init__(self, model, cfg):
        super(FullModel, self).__init__()
        self.model = model
        self.cfg = cfg
        # criterion
        if cfg.LOSS.USE_OHEM:
            self.Loss_seg = OhemCrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                            thres=cfg.LOSS.OHEMTHRES,
                                            min_kept=cfg.LOSS.OHEMKEEP,
                                            weight=None)
        else:
            self.Loss_seg = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,weight=None)

        self.Loss_keypoint = torch.nn.MSELoss(size_average=True)
        self.Loss_Line = torch.nn.MSELoss(size_average=True)
        # AutoWeightLoss
        if self.cfg.TRAIN.AUTOWEIGHTLOSS:
            self.AutoWeightLoss = AutomaticWeightedLoss(len(self.cfg.TRAIN.TASK)+len(self.cfg.TRAIN.ENHANCE))
        else:
            self.AutoWeightLoss = None


    def forward(self, inputs, labels, *args, **kwargs):
        masks, heatmaps, lines = labels
        model_outputs = self.model(inputs, *args, **kwargs)   # (B,5,130,130)
        seg_output, keypoint_output, line_output = model_outputs
        loss_seg = self.Loss_seg(seg_output, masks)       # label:(B,520,520)
        loss_keypoint = self.Loss_keypoint(keypoint_output, heatmaps)
        loss_line = self.Loss_Line(line_output, lines)
        loss_list = []
        loss_info= {
            'loss_type':[],
            'loss_effi':[],
            'loss_list':[]
        }
        if 'seg' in self.cfg.TRAIN.TASK:
            loss_list.append( loss_seg)
            loss_info['loss_list'].append(loss_seg.detach().cpu().item())
            loss_info['loss_type'].append('seg')
            loss_info['loss_effi'].append(self.cfg.TRAIN.SEG_EFFI)
        if 'keypoint' in self.cfg.TRAIN.TASK:
            loss_list.append( loss_keypoint )
            loss_info['loss_list'].append(loss_keypoint.detach().cpu().item())
            loss_info['loss_type'].append('keypoint')
            loss_info['loss_effi'].append(self.cfg.TRAIN.KEYPOINT_EFFI)
        if 'line' in self.cfg.TRAIN.TASK:
            loss_list.append( loss_line )
            loss_info['loss_list'].append(loss_line.detach().cpu().item())
            loss_info['loss_type'].append('line')
            loss_info['loss_effi'].append(self.cfg.TRAIN.LINE_EFFI)
        if 'seg-keypoint' in self.cfg.TRAIN.ENHANCE:
            loss_enhance = self.seg_keypoint_enhace(seg_output, keypoint_output)
            loss_list.append( loss_enhance )
            loss_info['loss_list'].append(loss_enhance.detach().cpu().item())
            loss_info['loss_type'].append('seg-keypoint')
            loss_info['loss_effi'].append(self.cfg.TRAIN.SEG_KEYPOINT_EFFI)
        if 'Keypoint-line' in self.cfg.TRAIN.ENHANCE:
            pass
        # 自动多任务权重
        if self.AutoWeightLoss is not None:
            loss_list,loss_effi = self.AutoWeightLoss(loss_list,loss_info['loss_effi'])
            loss_info['loss_effi'] = loss_effi
        loss_total = 0
        for loss in loss_list:
            loss_total += loss
        return torch.unsqueeze(loss_total,0), model_outputs , loss_info

    def seg_keypoint_enhace(self, seg_output, keypoint_output):
        mask = torch.argmax(seg_output.detach(),dim=1).cpu().numpy()
        generate_point = [mask2point(b_mask) for b_mask in mask]
        if None in generate_point:
            return torch.Tensor([0])
        else:
            heatmap = np.zeros(mask.shape)
            for b_id in range(mask.shape[0]):
                for p_id in range(3):
                    heatmap[b_id] = generate_heatmap(heatmap[b_id],generate_point[b_id][p_id], sigma=self.cfg.SIGMA)
            heatmap = torch.Tensor(heatmap).to(keypoint_output.device)
            loss_enhance = self.Loss_keypoint(keypoint_output, heatmaps)
            return loss_enhance


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, loss_list, loss_effi):
        for i, loss in enumerate(loss_list):
            loss_list[i] = 0.5 / (self.params[i] ** 2) * loss* loss_effi[i] + torch.log(1 + self.params[i] ** 2)
            loss_effi[i]*= 0.5 / (self.params[i] ** 2).detach().cpu().item()
        return loss_list, loss_effi


def mask2point(mask):
    point_list = np.zeros((6,2))
    # 获取4号点，位于区域3最下端和最右端的平均
    index = np.argwhere(mask==3)
    if len(index)==0:
        return None
    y1 = np.argmax(index[:,0])
    y2 = np.argmax(index[:,1])
    point_list[4-1] = np.round(((index[y1]+index[y2])/2))
    # 获取5号点，位于区域1最下端
    index = np.argwhere(mask==1)
    if len(index)==0:
        return None
    y = np.argmax(index[:,0])
    point_list[5-1] = index[y]
    # 获取6号点，位于区域4几何中心
    index = np.argwhere(mask==4)
    if len(index)==0:
        return None
    p = np.mean(index, axis=0)
    point_list[6-1] = np.round(p)
    return point_list[3:]


def generate_heatmap(img, pt, sigma, label_type='Gaussian'):
    # Check that any part of the gaussian is in-bounds
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    else:
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[1]), min(br[1], img.shape[1]) - ul[1]
    g_y = max(0, -ul[0]), min(br[0], img.shape[0]) - ul[0]
    # Image range
    img_x = max(0, ul[1]), min(br[1], img.shape[1])
    img_y = max(0, ul[0]), min(br[0], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())