# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Referring to the implementation in 
# https://github.com/zhanghang1989/PyTorch-Encoding
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import json
import torch
from torch.nn import functional as F
from PIL import Image

from .base_dataset import BaseDataset
from .utils import *

class HipJointDataset(BaseDataset):
    def __init__(self,
                 cfg,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=59,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(480, 480),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(HipJointDataset, self).__init__(cfg,ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.cfg = cfg
        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        # 关键点相关
        #self.heatmap_size = cfg.TRAIN.HEATMAP_SIZE
        #self.sigma = cfg.TRAIN.SIGMA

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, mask_path, point_path = item
            name = os.path.splitext(os.path.basename(mask_path))[0]
            sample = {
                'img': image_path,
                'label': mask_path,
                'keypoint':point_path,
                'name': name
            }
            files.append(sample)
        return files

    def load_keypoint(self,keypoint_path):
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)    # shape(6,3)
        keypoint = keypoint[:,[1,0]]
        return keypoint

    def keypoint2line(self,keypoint):
        # input : shape(6,2)
        # output : shape(3,2)   (半径,角度)
        points = keypoint.copy()[:,[1,0]]
        points = points.reshape(3,2,2)    # point2line函数的点是 (w,h)
        line_list = [list(point2line(ps[0], ps[1])) for ps in points ]
        return np.array(line_list)

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, item['img'])
        label_path = os.path.join(self.root, item['label'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        ) # (H,W,3)  (行数，列数，) 值域255
        # 读取mask
        label = np.array(
            Image.open(label_path).convert('P')
        )
        # 读取keypoint
        keypoint_path = os.path.join(self.root, item['keypoint'])
        keypoint = self.load_keypoint(keypoint_path)
        # 生成直线标签
        line = self.keypoint2line(keypoint)

        size = label.shape

        meta = {}
        meta['img_size'] = np.array(size)
        meta['name'] = name
        meta['scale'] = [1,1]
        meta['heatmap_size'] = self.heatmap_size
        meta['crop_locate'] = [0,0]
        meta['origin_line'] = line.copy()
        meta['origin_keypoint'] = keypoint.copy()

        if 'test' in self.list_path:
            image, meta = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8,
                return_padding=True,
                meta=meta
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), keypoint.copy(), line.copy(), meta

        if 'val' in self.list_path:
            image, label, keypoint, line, meta = self.resize_short_length(
                image,
                label=label,
                keypoint=keypoint,
                line = line,
                short_length=self.base_size,
                fit_stride=8,
                meta=meta
            )
            image, label, keypoint, line, meta = self.rand_crop(image, label, keypoint, line, meta)
            image = self.input_transform(image)
            heatmap = self.keypoint_transform(keypoint)
            line = self.line_transform(line)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), heatmap.copy(), line.copy(), meta

        # 等比例放缩
        image, label, keypoint, line, meta = self.resize_short_length(image, label, keypoint , line, short_length=self.base_size, meta=meta) 
        ''' 
        meta = {
            'scale':int,
            'crop_locate':list
            'padding':[],
            'name':std,
            'img_size':[]
        }
        '''
        image, label, heatmap, line, meta = self.gen_sample(image, label, keypoint, line, self.multi_scale, self.flip, meta=meta)

        return image.copy(), label.copy(), heatmap.copy(), line.copy(), meta