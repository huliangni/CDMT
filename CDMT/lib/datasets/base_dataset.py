# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import random
import math
import torch
from torch.nn import functional as F
from torch.utils import data

from config import config


class BaseDataset(data.Dataset):
    def __init__(self,
                 cfg,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.heatmap_size = cfg.TRAIN.HEATMAP_SIZE
        self.sigma = cfg.TRAIN.SIGMA

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')

    def keypoint_transform(self, keypoint):
        heatmap = np.zeros((keypoint.shape[0], self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        keypoint = keypoint/4
        keypoint = keypoint.astype(np.int32)
        for i in range(heatmap.shape[0]):
            heatmap[i] = self.generate_heatmap(heatmap[i], keypoint[i], self.sigma)   # 生成热力图
        meta = {
            'final_keypoint':keypoint
        }
        return heatmap

    def line_transform(self, line):
        line[:,0]/= self.base_size
        line[:,1]/= 180
        return line.astype(np.float32)

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label, keypoint, line, meta={}):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        # 裁剪考虑label的区域，避免裁剪掉label
        h_index, w_index = np.where(label>0)   # h_index, 
        h_min = max(h_index.min()-5,0)
        h_max = min(h_index.max()+5,new_h)
        w_min = max(w_index.min()-5,0)
        w_max = min(w_index.max()+5,new_w)
        # 裁剪的开始位置
        x = random.randint(max(0,w_max-self.crop_size[1]), min(w_min,new_w - self.crop_size[1]))
        y = random.randint(max(0,h_max-self.crop_size[0]), min(h_min,new_h - self.crop_size[0]))
        #x = random.randint(0, new_w - self.crop_size[1])
        #y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]] 
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        keypoint[:,0] = keypoint[:,0]-y
        keypoint[:,1] = keypoint[:,1]-x
        line[:,0] = line[:,0] - y*np.sin(line[:,1]/180*np.pi) - x*np.cos(line[:,1]/180*np.pi)
        #assert keypoint.min() >= 0, meta['name']
        #assert keypoint.max()<self.crop_size[0], meta['name']
        meta['crop_locate'] = [y,x]
        return image, label, keypoint, line, meta

    def generate_heatmap(self,img, pt, sigma, label_type='Gaussian'):
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

        '''
        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])
        '''
        # Usable gaussian range
        g_x = max(0, -ul[1]), min(br[1], img.shape[1]) - ul[1]
        g_y = max(0, -ul[0]), min(br[0], img.shape[0]) - ul[0]
        # Image range
        img_x = max(0, ul[1]), min(br[1], img.shape[1])
        img_y = max(0, ul[0]), min(br[0], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

    def multi_scale_aug(self, image, label=None, keypoint=None, line=None,
                        rand_scale=1, rand_crop=True, meta={}):
        long_size = np.int32(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int32(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int32(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)

        h_scale = new_h/h
        w_scale = new_w/w
        if keypoint is not None:
            keypoint[:,0]*=h_scale
            keypoint[:,1]*=w_scale

        if line is not None:
            line[:,0] *= (h_scale+w_scale)/2

        meta['scale'] = [ meta['scale'][0]*h_scale, meta['scale'][1]*w_scale ]

        if (label is None) and (keypoint is None) and (line is None):
            return image, meta

        if rand_crop:
            image, label, keypoint, line, meta = self.rand_crop(image, label, keypoint, line, meta=meta)

        return image, label, keypoint, line, meta

    def resize_short_length(self, image, label=None, keypoint=None , line=None,  short_length=None, fit_stride=None, return_padding=False, meta={}):
        h, w = image.shape[:2]
        if h < w:   # 按照最短边来等比例放缩
            new_h = short_length
            new_w = np.int32(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = np.int32(h * short_length / w + 0.5)    
        
        image = cv2.resize(image, (new_w, new_h),       # cv2.resize是先w后h，而img是先h再w
                           interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=tuple(x * 255 for x in self.mean[::-1])
            )

        if label is not None:
            label = cv2.resize(
                label, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w,  # 上下左右
                    cv2.BORDER_CONSTANT, value=self.ignore_label
                )

        h_scale = new_h/h
        w_scale = new_w/w
        if keypoint is not None:
            keypoint[:,0]*=h_scale
            keypoint[:,1]*=w_scale

        if line is not None:
            line[:,0]*= (h_scale+w_scale)/2

        meta['scale'] = [meta['scale'][0]*h_scale,meta['scale'][1]*w_scale]
        meta['padding'] = [pad_h,pad_w]

        '''
        if (label is not None) or (keypoint is not None):
            if return_padding:
                return image, label, keypoint, (pad_h, pad_w)
            else:
                return image, label, keypoint
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image,
        '''
        if (label is not None) or (keypoint is not None) or (line is not None):
            return image, label, keypoint, line, meta
        else:
            return image, meta

    def random_brightness(self, img):
        if not config.TRAIN.RANDOM_BRIGHTNESS:
            return img
        if random.random() < 0.5:
            return img
        self.shift_value = config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def get_label_sclae(self,label):
        # 目的：放缩后的区域，最长核心边界，不能超过裁剪后的大小（520）
        # label: shape(h,w)
        h_index, w_index = np.where(label>0)   # h_index, 
        h_min = h_index.min()
        h_max = h_index.max()
        w_min = w_index.min()
        w_max = w_index.max()
        # 计算不超过边界的最大放缩尺度
        max_size  = max((w_max-w_min),(h_max-h_min))
        max_scale = self.base_size/max_size
        return max_scale    # 留一些余裕

    def gen_sample(self, image, label, keypoint, line,
                   multi_scale=True, is_flip=True, meta={}):
        if multi_scale:     # 不同尺度放缩+crop
            label_scale = self.get_label_sclae(label)
            max_scale = min(int((label_scale*0.9-0.5)*10), self.scale_factor)
            rand_scale = 0.5 + random.randint(0, max_scale) / 10.0  # 0.5~2.0
            image, label, keypoint, line, meta = self.multi_scale_aug(image, label,keypoint, line,
                                                rand_scale=rand_scale, meta=meta)

        image = self.random_brightness(image)
        image = self.input_transform(image)
        label = self.label_transform(label)
        heatmap = self.keypoint_transform(keypoint)
        line = self.line_transform(line)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label, heatmap, line, meta

    def reduce_zero_label(self, labelmap):
        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1

        return encoded_labelmap

    def decode_line(self, line, meta):
        device = line.device
        line = line.cpu()
        line[:,:,1]*= 180
        line[:,:,0]*= meta['heatmap_size'][0][:,None]*4
        y = meta['crop_locate'][0][:,None]
        x = meta['crop_locate'][1][:,None]
        line[:,:,0] = line[:,:,0] + y*np.sin(line[:,:,1]/180*np.pi) + x*np.cos(line[:,:,1]/180*np.pi)
        line[:,:,0]/= (meta['scale'][0]+meta['scale'][1])[:,None]/2
        return line.to(device)

    def decode_heatmap(self,heatmap,meta):
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
            
            for n in range(coords.size(0)):
                for p in range(coords.size(1)):
                    hm = heatmap[n][p]
                    px = int(math.floor(coords[n][p][1]))
                    py = int(math.floor(coords[n][p][0]))
                    '''
                    if (px > 1) and (px < hm.shape[0]) and (py > 1) and (py < hm.shape[1]):
                        diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                        coords[n][p] += diff.sign() * .25
                    '''
                    '''
                    if (px > 1) and (px < hm.shape[0]-1) and (py > 1) and (py < hm.shape[1]-1):
                        diff = torch.Tensor([hm[py ][px+1] - hm[py ][px - 1], hm[py+1][px ]-hm[py - 1][px ]])
                        coords[n][p] += diff.sign() * .25
                    '''
            coords += 0.5
            return coords

        coords = heatmap2coord(heatmap)
        coords = coords.cpu()
        preds = pose_process(heatmap, coords).clone()
        keypoint = preds*4
        keypoint[:,:,0]+= meta['crop_locate'][0][:,None]
        keypoint[:,:,1]+= meta['crop_locate'][1][:,None]
        keypoint[:,:,0]/=meta['scale'][0][:,None]
        keypoint[:,:,1]/=meta['scale'][1][:,None]
        return torch.round(keypoint).to(heatmap.device)


    def inference(self, config, model, image, flip=False, meta={}):
        size = image.size()
        pred_mask, pred_heatmap, pred_line = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred_mask = pred_mask[config.TEST.OUTPUT_INDEX]

        pred_mask = F.interpolate(
            input=pred_mask, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        pred_keypoint = self.decode_heatmap(pred_heatmap, meta)

        pred_line = self.decode_line(pred_line, meta)

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred_mask += flip_pred
            pred_mask = pred_mask * 0.5
        return pred_mask.exp(), pred_keypoint, pred_line

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False, meta={}):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int32(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int32(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        padvalue = -1.0 * np.array(self.mean) / np.array(self.std)
        for scale in scales: # 注意不同scale的meta
            # 输入的Image最短边是520,变成最长边为scale*520
            new_img, meta = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False,
                                           meta=meta)
            height, width = new_img.shape[:-1]  # 填补前的形状

            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width,
                                         self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img) # shape(1,3,520,520)
                pred_mask, pred_keypoint, pred_line = self.inference(config, model, new_img, flip, meta=meta)    
                # pred_keypoint已经是恢复到原始图像的size
                # pred_mask = (1,5,520,520)  已经从130插值回来了
                # pred_line 已经是完全恢复
                pred_mask = pred_mask[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width,
                                             self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:-1]
                rows = np.int32(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int32(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                pred_mask = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img,
                                                      h1-h0,
                                                      w1-w0,
                                                      self.crop_size,
                                                      padvalue)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        pred_mask[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1-h0, 0:w1-w0]
                        count[:, :, h0:h1, w0:w1] += 1
                pred_mask = pred_mask / count
                pred_mask = pred_mask[:, :, :height, :width]

            pred_mask = F.interpolate(
                pred_mask, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            '''
            pred_keypoint[:,:,0] = pred_keypoint[:,:,0]*ori_height/height
            pred_keypoint[:,:,1] = pred_keypoint[:,:,1]*ori_width/width
            '''

            #final_pred += pred_mask
        return pred_mask,pred_keypoint, pred_line
