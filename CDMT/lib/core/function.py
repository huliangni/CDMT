# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import decode_heatmap, decode_keypoint, decode_line
from utils.plot import plot_multi_mask

import utils.distributed as dist


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict, device):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    record_loss_list = [ AverageMeter() for _ in range(len(config.TRAIN.TASK)+len(config.TRAIN.ENHANCE)) ]
    record_loss_effi = [ AverageMeter() for _ in range(len(config.TRAIN.TASK)+len(config.TRAIN.ENHANCE)) ]
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, masks, heatmaps, lines, meta = batch
        images = images.to(device)
        masks = masks.long().to(device)
        heatmaps = heatmaps.to(device)
        lines = lines.to(device)

        losses, _ , loss_info = model(images, [masks, heatmaps, lines])
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        for n_task in range((len(config.TRAIN.TASK)+len(config.TRAIN.ENHANCE))):
            record_loss_list[n_task].update(loss_info['loss_list'][n_task])
            record_loss_effi[n_task].update(loss_info['loss_effi'][n_task])


        if config.TRAIN.LR_ADJAST == 'exp':
            lr = adjust_learning_rate(optimizer,
                                    base_lr,
                                    num_iters,
                                    i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train/train_loss', ave_loss.average(), global_steps)
    for task_id, task_name in enumerate(loss_info['loss_type']):
        writer.add_scalar(f'train/loss_{task_name}', record_loss_list[task_id].average(), global_steps)
        writer.add_scalar(f'train/effi_{task_name}', record_loss_effi[task_id].average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict, device):
    model.eval()
    save_dict = {}
    ave_loss = AverageMeter()
    ave_keypoint_error1 = AverageMeter()
    ave_keypoint_error2 = AverageMeter()
    ave_line_error = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            images, masks, heatmaps, lines, meta = batch
            size = masks.size()
            images = images.to(device)
            masks = masks.long().to(device)
            heatmaps = heatmaps.to(device)
            lines = lines.to(device)

            losses, preds, loss_info = model(images, [masks, heatmaps, lines])

            pred_mask, pred_heatmap, pred_line = preds

            # 分割后处理
            if not isinstance(pred_mask, (list, tuple)):
                pred_mask = [pred_mask]
            for i, x in enumerate(pred_mask):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    masks,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            # 关键点后处理1 heatmap->低分辨率关键点
            label_keypoint1 = decode_heatmap(heatmaps.cpu())
            pred_keypoint1 = decode_heatmap(pred_heatmap.cpu())
            point_error1 = torch.abs(label_keypoint1-pred_keypoint1).mean().item()
            ave_keypoint_error1.update(point_error1)
            # 关键点后处理2 低分辨率关键点->原图的关键点
            label_keypoint2 = torch.round( decode_keypoint(label_keypoint1.clone(), meta) )
            pred_keypoint2 = torch.round( decode_keypoint(pred_keypoint1.clone(), meta) )
            point_error2 = torch.abs(label_keypoint2-pred_keypoint2).mean().item()
            ave_keypoint_error2.update(point_error2)
            # 关键点后处理complex 低分辨率关键点->原图的关键点，用插值的方式
            #label_keypoint_complex = decode_keypoint_complex(label_keypoint1.clone(), meta)
            #pred_keypoint_complex = decode_keypoint_complex(pred_keypoint1.clone(), meta)
            #point_error_complex = torch.abs(label_keypoint_complex-pred_keypoint_complex).mean().item()
            #ave_keypoint_error_complex.update(point_error_complex)

            # 直线后处理
            pred_line = decode_line(pred_line.cpu(), meta)
            label_line = decode_line(lines.cpu(), meta)
            ave_line_error.update((torch.abs(pred_line-label_line)[:,:,1]).mean().item())

            for i in range(len(meta['name'])):
                name = '_'.join(meta['name'][i].split('_')[:-1])
                save_dict[name] = {
                    'pred_mask':torch.argmax(pred_mask[0][i],dim=0).cpu().numpy(),
                    'pred_keypoint': torch.round(pred_keypoint2)[i].cpu().numpy(),
                    'pred_line':pred_line[i].cpu().numpy()
                }

            if idx % 10 == 0:
                print(idx)

            # 这个loss包含了autoweighted的正则
            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            # 去掉正则 
            #valid_loss = 0
            #for n_task in range(len(config.TRAIN.TASK)):
            #    valid_loss += loss_info['loss_list'][n_task]*loss_info['loss_effi'][n_task]
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    joint_metric = 0
    joint_metric+= 5-ave_keypoint_error2.average()
    joint_metric+= (mean_IoU-0.7)*10
    joint_metric+= 3-ave_line_error.average()

    print(f'error 1 {ave_keypoint_error1.average()}, error 2 {ave_keypoint_error2.average()}')
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid/valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid/joint_metric', joint_metric, global_steps)
    writer.add_scalar('valid/valid_mIoU', mean_IoU, global_steps)
    writer.add_scalar('valid/point_error1', ave_keypoint_error1.average(), global_steps)
    writer.add_scalar('valid/point_error2', ave_keypoint_error2.average(), global_steps)
    writer.add_scalar('valid/line_error', ave_line_error.average(), global_steps)
    #writer.add_scalar('valid_point_error_complex', ave_keypoint_error_complex.average(), global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), joint_metric,  save_dict


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    save_dict = {}
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label_mask, label_keypoint, line, meta = batch
            size = label_mask.size()
            pred_mask, pred_keypoint, pred_line = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST,
                meta = meta)

            # pred_mask是第一次long_resize后的形状,shape=(520,520+)，最短边是520
            # pred_keypoint是原始图像是size
            # pred_line也是原始图像size

            ori_img = image.clone()
            if len(meta['padding']) > 0:
                border_padding = meta['padding']
                pred_mask = pred_mask[:, :, 0:pred_mask.size(2) - border_padding[0], 0:pred_mask.size(3) - border_padding[1]]
                ori_img = image[:, :, 0:pred_mask.size(2) - border_padding[0], 0:pred_mask.size(3) - border_padding[1]]

            if pred_mask.size()[-2] != size[-2] or pred_mask.size()[-1] != size[-1]:
                #pred_keypoint[:,:,0] = pred_keypoint[:,:,0]/pred_mask.shape[-2]*size[-2]
                #pred_keypoint[:,:,1] = pred_keypoint[:,:,1]/pred_mask.shape[-1]*size[-1]
                pred_mask = F.interpolate(
                    pred_mask, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                ori_img = F.interpolate(
                    ori_img, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label_mask,
                pred_mask,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            '''
            for i in range(len(name)):
                plot_multi_mask(ori_img[i].cpu().numpy(), torch.argmax(pred_mask[i],dim=0).cpu().numpy(), 'data/HipJoint/pred_plot/'+name[i]+'.png', scale=True)
                plot_multi_mask(ori_img[i].cpu().numpy(), label_mask[i].cpu().numpy(), 'data/HipJoint/mask_plot/'+name[i]+'.png', scale=True)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))
            '''
            for i in range(len(meta['name'])):
                name = '_'.join(meta['name'][i].split('_')[:-1])
                save_dict[name] = {
                    'pred_mask':torch.argmax(pred_mask[i],dim=0).cpu().numpy(),
                    'pred_keypoint': torch.round(pred_keypoint)[i].cpu().numpy(),
                    'pred_line' : pred_line[i].cpu().numpy()
                }


    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return save_dict


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
