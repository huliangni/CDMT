import os
import random
import math
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import json

'''
def angle2type(alpha,beta):
    if alpha >=60:
        if beta<=55:
            return 'Ia'
        else:
            return 'Ib'
    elif alpha >=50:
        return 'IIa'
    elif alpha >= 43 :
        if beta <= 77:       
            return 'IIc'
        else:
            return 'D'
    else :
        return 'III'
'''
def angle2type(alpha,beta):
    if alpha >=60:
        if beta<=55:
            return 'I'
        else:
            return 'I'
    elif alpha >=50:
        return 'II'
    elif alpha >= 43 :
        if beta <= 77:       
            return 'II'
        else:
            return 'D'
    else :
        return 'III'

def IOU_metric(pred,label, class_num):
    '''
        pred, label : shape(H,W) , number 0~4
    '''
    IOU_list = []
    for i in range(1,class_num):
        index_pred = (pred==i)
        index_label = (label==i)
        IOU_list.append( (index_pred&index_label).sum() / ((index_pred|index_pred).sum()+1e-5) )
    return IOU_list

def DSC_metric(pred,label, class_num):
    '''
        pred, label : shape(H,W) , number 0~4
    '''
    DSC_list = []
    for i in range(1,class_num):
        index_pred = (pred==i)
        index_label = (label==i)
        DSC_list.append( 2*(index_pred&index_label).sum() / (index_pred.sum()+1e-5+index_label.sum()) )
    return DSC_list

def read_files(file_list):
    files = []
    for item in file_list:
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

def mask2point(mask):
    point_list = np.zeros((6,2))
    # 获取1号点，用2号区域的最左端代替
    if np.sum(mask==2) > 0:
        index = np.argwhere(mask==2)
    else:
        index = np.argwhere(mask==0)
    y = np.argmin(index[:,1])
    point_list[1-1] = index[y]
    # 获取2号点，用1号点平移代替
    point_list[2-1][0] = point_list[1-1][0].item()
    point_list[2-1][1] = point_list[1-1][1].item()+20
    # 获取3号点，用1号区域的最右端代替
    if np.sum(mask==1) > 0:
        index = np.argwhere(mask==1)
    else:
        index = np.argwhere(mask==0)
    y = np.argmax(index[:,1])
    point_list[3-1] = index[y]
    
    # 获取4号点，位于区域3最下端和最右端的平均
    if np.sum(mask==3) > 0:
        index = np.argwhere(mask==3)
    else:
        index = np.argwhere(mask==0)
    y1 = np.argmax(index[:,0])
    y2 = np.argmax(index[:,1])
    point_list[4-1] = np.round(((index[y1]+index[y2])/2))
    '''
    # 获取4号点，位于区域3最右端
    if np.sum(mask==3) > 0:
        index = np.argwhere(mask==3)
    else:
        index = np.argwhere(mask==0)
    y2 = np.argmax(index[:,1])
    point_list[4-1] = index[y2]
    '''
    # 获取5号点，位于区域1最下端
    if np.sum(mask==1) > 0:
        index = np.argwhere(mask==1)
    else:
        index = np.argwhere(mask==0)
    index = np.argwhere(mask==1)
    y = np.argmax(index[:,0])
    point_list[5-1] = index[y]
    # 获取6号点，位于区域4几何中心
    if np.sum(mask==4) > 0:
        index = np.argwhere(mask==4)
    else:
        index = np.argwhere(mask==0)
    p = np.mean(index, axis=0)
    point_list[6-1] = np.round(p)
    return point_list

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

def points2angle(points):
    points = points.reshape(3,2,2)
    line_list = [point2line(ps[0], ps[1]) for ps in points ]
    theta_list = [line[1] for line in line_list ]
    alpha,beta = line2angle(*tuple(theta_list))
    return [alpha,beta]

def line2angle(theta1,theta2,theta3):
    # 这个theta是直线的切点的角度，相当于从上y轴开始顺时针
    theta1 = 90
    alpha = theta2-theta1
    beta = theta1-theta3
    return alpha,beta

def calibration_mask_point(data_dir):
    # 根据分割标注，来生成关键点3~6号。关键点1~2号的x坐标取平均

    file_list = [line.strip().split() for line in open(data_dir)]
    files = read_files(file_list)

    for id,item in enumerate(files):
        name = '_'.join((item["name"].split('_')[:-1]))
        image_path = item['img'].replace('..','data')
        mask_path = item['label'].replace('..','data')
        keypoint_path = item['keypoint'].replace('..','data')

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        label_mask = np.array(
            Image.open(mask_path).convert('P')
        )

        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)[:,:2]    # shape(6,3)

        # 分割提取出点，与预测关键点增强
        seg_point = mask2point(label_mask)
        seg_point = seg_point[:,[1,0]]      # 存储的是[x,y]

        calib_point = np.zeros(seg_point.shape)
        calib_point[:2] = keypoint[:2]
        calib_point[:2,1] = np.round(np.mean(calib_point[:2,1]))
        calib_point[2:] = seg_point[2:]

        # 生成保存格式
        calib_point_dict = { str(i):calib_point[i-1].tolist()  for i in range(1,7) }
        with open(keypoint_path.replace('annotations', 'annotations_calibration'), 'w') as json_file:
            json.dump(calib_point_dict, json_file, indent=4)

def clean_data(data_dir, pred_dir, task_list=[]):
    error_list, disease_list = post_angle_loss(data_dir, pred_dir, task_list)
    error_thresh = np.arange(3,11)
    for thresh in error_thresh:
        index = error_list.max(1)<thresh
        print('error thresh ',thresh,' 角度误差 ',error_list[index].mean(), '疾病准确率 ', disease_list[index].mean() ,'去除数量 ',index.sum())
    # 下面开始清洗
    clean_list = []
    error_thresh = 5
    index = error_list.max(1)<error_thresh
    file_list = [line.strip() for line in open(data_dir)]
    assert len(index) == len(file_list)
    for i in range(len(index)):
        if index[i]==True:
            clean_list.append(file_list[i])
    with open(os.path.join('data/HipJoint','fullset_clean.txt'), 'w') as file:
        for line in clean_list:
            file.write(line)
            file.write("\n")
    split_dataset(clean_list)

def split_dataset(clean_list,root='data/HipJoint'):
    split_index = np.arange(len(clean_list))
    np.random.shuffle(split_index)
    train_index = split_index[:int(0.7*len(clean_list))]
    valid_index = split_index[int(0.7*len(clean_list)):int(0.8*len(clean_list))]
    test_index = split_index[int(0.8*len(clean_list)):]
    # 写入划分后的数据集
    with open(os.path.join(root,'train_clean.txt'), 'w') as file:
        for id in train_index:
            file.write(clean_list[id])
            file.write("\n")
    with open(os.path.join(root,'valid_clean.txt'), 'w') as file:
        for id in valid_index:
            file.write(clean_list[id])
            file.write("\n")
    with open(os.path.join(root,'test_clean.txt'), 'w') as file:
        for id in test_index:
            file.write(clean_list[id])
            file.write("\n")


def merge_angle(angle_list,thresh, best_id):
    if len(angle_list) == 1:
        return angle_list[0]
    elif len(angle_list) == 2:
        if np.abs(angle_list[0]-angle_list[1])<thresh:
            return np.mean(angle_list)
        else:
            return angle_list[best_id]
    elif len(angle_list) == 3:
        diff01 = np.abs(angle_list[0]-angle_list[1])
        diff02 = np.abs(angle_list[0]-angle_list[2])
        diff12 = np.abs(angle_list[2]-angle_list[1])
        if diff01>thresh and diff02 > thresh and diff12>thresh:
            return angle_list[best_id]
        elif diff01<thresh and diff02>thresh and diff12>thresh:
            return np.mean(angle_list[[0,1]])
        elif diff01>thresh and diff02<thresh and diff12>thresh:
            return np.mean(angle_list[[0,2]])
        elif diff01>thresh and diff02>thresh and diff12<thresh:
            return np.mean(angle_list[[2,1]])
        elif diff01<thresh and diff02<thresh and diff12<thresh:
            return np.mean(angle_list)
        else:
            return np.mean(angle_list)

def post_angle_loss(data_dir, pred_dir, task_list=[]):
    label_angle_list = []
    pred_angle_list = []
    save_dict = {}
    disease_rate = []

    file_list = [line.strip().split() for line in open(data_dir)]
    files = read_files(file_list)

    pred_file= torch.load(pred_dir)

    for id,item in enumerate(files):
        name = '_'.join((item["name"].split('_')[:-1]))
        image_path = item['img'].replace('..','data')
        mask_path = item['label'].replace('..','data')
        keypoint_path = item['keypoint'].replace('..','data')
        # 加载标签
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        label_mask = np.array(
            Image.open(mask_path).convert('P')
        )
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)[:,:2]    # shape(6,3)
        label_angle = points2angle(keypoint)

        # 处理多个任务转化为角度
        pred_angle = []
        if 'seg' in task_list:
            pred_mask = pred_file[name]['pred_mask']
            pred_seg_point = mask2point(pred_mask)
            pred_seg_angle = points2angle(pred_seg_point[:,[1,0]])
            pred_angle.append(pred_seg_angle)
        if 'keypoint' in task_list:
            pred_keypoint = pred_file[name]['pred_keypoint']
            pred_keypoint_angle = points2angle(pred_keypoint[:,[1,0]])
            pred_angle.append(pred_keypoint_angle)
        if 'line' in task_list:
            pred_line = pred_file[name]['pred_line']
            pred_theta = [line[1] for line in pred_line ]
            pred_line_angle = line2angle(*tuple(pred_theta))
            pred_angle.append(pred_line_angle)

        # 角度融合
        pred_angle = np.array(pred_angle)
        best_beta_id = -1 # 优先Line, 无则keypoint
        if 'keypoint' in task_list:
            best_alpha_id = task_list.index('keypoint')
        else:
            best_alpha_id = -1
        merge_alpha = merge_angle(pred_angle[:,0], thresh=3, best_id=best_alpha_id)
        merge_beta = merge_angle(pred_angle[:,1], thresh=5, best_id=best_beta_id)
        #merge_beta = np.mean(pred_angle[[1,2],1])
        pred_angle = [merge_alpha,merge_beta]
        label_angle_list.append(label_angle)
        pred_angle_list.append(pred_angle)

        disease_rate.append(angle2type(*tuple(label_angle)) == angle2type(*tuple(pred_angle)))

        save_dict[name] = {}

    label_angle_list = np.array(label_angle_list)
    pred_angle_list = np.array(pred_angle_list)

    error = np.abs(label_angle_list-pred_angle_list)

    disease_rate = np.array(disease_rate)

    print("角度误差",error.mean(0), error.std(0),"  疾病准确率",disease_rate.mean(0))
    return error, disease_rate

def seg_angle_loss(data_dir, pred_dir):
    label_angle_list = []
    pred_angle_list = []
    save_dict = {}
    disease_rate = []
    IOU_multi_list = []
    DSC_multi_list = []

    file_list = [line.strip().split() for line in open(data_dir)]
    files = read_files(file_list)

    pred_file= torch.load(pred_dir)

    for id,item in enumerate(files):
        name = '_'.join((item["name"].split('_')[:-1]))
        image_path = item['img'].replace('..','data')
        mask_path = item['label'].replace('..','data')
        keypoint_path = item['keypoint'].replace('..','data')

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        label_mask = np.array(
            Image.open(mask_path).convert('P')
        )
        pred_mask = pred_file[name]['pred_mask']

        IOU = IOU_metric(pred_mask, label_mask, class_num=5)
        IOU_multi_list.append(IOU)

        DSC = DSC_metric(pred_mask, label_mask, class_num=5)
        DSC_multi_list.append(DSC)

        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)[:,:2]    # shape(6,3)

        pred_point = mask2point(pred_mask)
        pred_point = pred_point[:,[1,0]]
        #pred_point[3:] = keypoint[3:]

        label_angle = points2angle(keypoint)
        pred_angle = points2angle(pred_point)

        label_angle_list.append(label_angle)
        pred_angle_list.append(pred_angle)

        disease_rate.append(angle2type(*tuple(label_angle)) == angle2type(*tuple(pred_angle)))

        save_dict[name] = {}
    

    print('每个结构IOU : ',np.mean(IOU_multi_list,axis=0),' 平均IOU : ',np.mean(IOU_multi_list))

    print('每个结构DSC : ',np.mean(DSC_multi_list,axis=0),' 平均DSC : ',np.mean(DSC_multi_list))

    label_angle_list = np.array(label_angle_list)
    pred_angle_list = np.array(pred_angle_list)

    error = np.abs(label_angle_list-pred_angle_list)
    disease_rate = np.array(disease_rate)
    
    print("角度误差",error.mean(0), error.std(0), " 疾病准确率",disease_rate.mean(0))

def point_angle_loss(data_dir, pred_dir):
    label_angle_list = []
    pred_angle_list = []
    point_error_list = []
    save_dict = {}
    disease_rate = []

    file_list = [line.strip().split() for line in open(data_dir)]
    files = read_files(file_list)

    pred_file= torch.load(pred_dir)

    for id,item in enumerate(files):
        name = '_'.join((item["name"].split('_')[:-1]))
        image_path = item['img'].replace('..','data')
        keypoint_path = item['keypoint'].replace('..','data')
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)[:,:2]    # shape(6,3)

        pred_point = pred_file[name]['pred_keypoint']
        pred_point = pred_point[:,[1,0]]
        point_error_list.append(np.abs(pred_point-keypoint))

        label_angle = points2angle(keypoint)
        pred_angle = points2angle(pred_point)

        label_angle_list.append(label_angle)
        pred_angle_list.append(pred_angle)

        disease_rate.append(angle2type(*tuple(label_angle)) == angle2type(*tuple(pred_angle)))

        save_dict[name] = {}
    
    label_angle_list = np.array(label_angle_list)
    pred_angle_list = np.array(pred_angle_list)

    error = np.abs(label_angle_list-pred_angle_list)
    disease_rate = np.array(disease_rate)
    #print(error.mean(0))
    for i in range(len(error)):
        #print(files[i]['name'],f'  alpha {error[i][0]} , beta {error[i][1]}')
        pass
    point_error_list = np.stack(point_error_list,axis=0)
    print('点坐标误差 ',point_error_list.mean(0).mean(-1))
    print("角度误差",error.mean(0), error.std(0),"  疾病准确率",disease_rate.mean(0))

def line_angle_loss(data_dir, pred_dir):
    label_angle_list = []
    pred_angle_list = []
    point_error_list = []
    save_dict = {}
    disease_rate = []

    file_list = [line.strip().split() for line in open(data_dir)]
    files = read_files(file_list)

    pred_file= torch.load(pred_dir)

    for id,item in enumerate(files):
        name = '_'.join((item["name"].split('_')[:-1]))
        image_path = item['img'].replace('..','data')
        keypoint_path = item['keypoint'].replace('..','data')
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)[:,:2]    # shape(6,3)

        keypoint = keypoint.reshape(3,2,2)
        label_line = [point2line(ps[0], ps[1]) for ps in keypoint ]
        pred_line = pred_file[name]['pred_line']

        label_theta = [line[1] for line in label_line ]
        label_angle = line2angle(*tuple(label_theta))
        pred_theta = [line[1] for line in pred_line ]
        pred_angle = line2angle(*tuple(pred_theta))

        label_angle_list.append(list(label_angle))
        pred_angle_list.append(list(pred_angle))

        disease_rate.append(angle2type(*tuple(label_angle)) == angle2type(*tuple(pred_angle)))

        save_dict[name] = {}
    
    label_angle_list = np.array(label_angle_list)
    pred_angle_list = np.array(pred_angle_list)

    error = np.abs(label_angle_list-pred_angle_list)
    disease_rate = np.array(disease_rate)
    print("角度误差",error.mean(0),error.std(0),"  疾病准确率",disease_rate.mean(0))




if __name__ == "__main__":

    #data_dir = 'data/HipJoint/test_all.txt'
    #calibration_mask_point(data_dir)



    data_dir = 'data/HipJoint/test_fullset.txt'
    pred_dir = 'output/hipjoint/w18_520_seg_keypoint_line_adam_exp_1916_calib_V0/predictions.pth'
    pred_dir = ''

    task_full_list = ['seg','keypoint','line']
    task_list = []
    for key in task_full_list:
        if key in pred_dir:
            task_list.append(key)

    if 'keypoint' in task_list:
        print('关键点结果: ')
        point_angle_loss(data_dir=data_dir, pred_dir=pred_dir)
    if 'seg' in task_list:
        print('分割结果:')
        seg_angle_loss(data_dir=data_dir, pred_dir=pred_dir)
    if 'line' in task_list :
        print('线回归结果: ')
        line_angle_loss(data_dir=data_dir, pred_dir=pred_dir)
    if len(task_list)>0:
        print('增强结果:')
        post_angle_loss(data_dir=data_dir, pred_dir=pred_dir, task_list=task_list)


    pred_dir = 'output/hipjoint/w18_520_seg_keypoint_line_adam_exp_1916_calib_V0/predictions.pth'

    task_full_list = ['seg','keypoint','line']
    task_list = []
    for key in task_full_list:
        if key in pred_dir:
            task_list.append(key)

    clean_data(data_dir, pred_dir, task_list=task_list)