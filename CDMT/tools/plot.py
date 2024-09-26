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
from pathlib import Path
from utils import *


def plot_keypoint_label():

    file_list = [line.strip().split() for line in open('data/HipJoint/test.txt')]
    files = read_files(file_list)

    for item in files:
        name = item["name"]
        image_path = item['img'].replace('..','data')
        keypoint_path = item['keypoint'].replace('..','data')
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)[:,:2]    # shape(6,3)

        # 绘制点
        for point in keypoint:
            cv2.circle(image, point.astype(np.int32), 5, (0, 0, 255), -1)  # 绘制红色点，半径为5

        save_dir = 'output/HipJoint/plot_label/'+name.replace('mask', 'keypoint_label')+'.png'
        cv2.imwrite(save_dir,image)


def plot_pred_point(data_dir='', with_mask=False):
    result_dict = torch.load(data_dir)
    save_dir = Path(data_dir).parent/'plot_mask_keypoint_pred'
    save_dir.mkdir(parents=True, exist_ok=True)
    error = []
    for name,item in result_dict.items():
        name
        image_path = 'data/HipJoint/images/'+name+'.jpg'
        mask_path = 'data/HipJoint/masks/'+name+'_mask.png'
        keypoint_path = 'data/HipJoint/annotations/'+name+'.json'
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        
        mask = np.array(
            Image.open(mask_path).convert('P')
        )
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2]) # shape(6,3)
        keypoint = np.round(keypoint)[:,:2]    # shape(6,2)
        
        mask_pred = item['pred_mask']
        keypoint_pred = item['pred_keypoint']
        keypoint_pred = keypoint_pred[:,[1,0]]
        error.append(np.abs(keypoint-keypoint_pred).mean())

        # 绘制mask
        if with_mask:
            image = _plot_mask_(image, mask)
        # 绘制点
        image = _plot_point_(image, keypoint_pred)

        cv2.imwrite(str(save_dir/(name+'.png')),image)
    print(np.mean(error))


def plot_line_label():

    file_list = [line.strip().split() for line in open('data/HipJoint/test.txt')]
    files = read_files(file_list)

    for item in files:
        name = item["name"]
        image_path = item['img'].replace('..','data')
        keypoint_path = item['keypoint'].replace('..','data')
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)[:,:2]    # shape(6,3)

        for line in keypoint.reshape(3,2,2):
            rho,theta_degrees = point2line(line[0],line[1])
            # 将极角从度转换为弧度
            theta_radians = np.radians(theta_degrees)

            # 计算直线的极坐标表示
            a = int(rho * np.cos(theta_radians))    # 切点位置极坐标转xy
            b = int(rho * np.sin(theta_radians))
            x2 = int(a + 1000 * (np.sin(theta_radians)))  # 1000是一个足够长的线段
            y2 = int(b + 1000 * (-np.cos(theta_radians)))
            x1 = int(a - 1000 * (np.sin(theta_radians)))  # 1000是一个足够长的线段
            y1 = int(b - 1000 * (-np.cos(theta_radians)))

            # 绘制直线
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # (0, 255, 0) 是颜色，2 是线宽

            #cv2.line(image, (100, 50), (400, 50), (0, 0, 255), 2)  # (0, 255, 0) 是颜色，2 是线宽
            '''
            画线说明：
            两点坐标: (x1,y1), (x2,y2) x是点的横坐标, y是点的纵坐标, 从左上角开始计数
            '''

        # 绘制点
        for point in keypoint:
            cv2.circle(image, point.astype(np.int32), 5, (0, 0, 255), -1)  # 绘制红色点，半径为5

        save_dir = 'output/HipJoint/plot_line_label/'+name.replace('mask', 'line_label')+'.png'
        cv2.imwrite(save_dir,image)


def plot_multi_mask(image,pred_mask, save_dir, scale=False):

    # 假设 mask_rcnn_results 是 Mask R-CNN 的输出，包含类别标签和分割掩模
    '''
    pred_mask : {key:value}  key:label_id  value:mask
    '''

    # 创建颜色映射字典，将每个类别映射到不同的颜色
    color_mapping = {
        1: (0, 0, 255),  # 红色
        2: (0, 255, 0),  # 绿色
        3: (255, 0, 0),  # 蓝色
        4: (255, 255, 0)  # 黄色
    }

    mean= np.array([0.485, 0.456, 0.406])[:,None,None]
    std= np.array([0.229, 0.224, 0.225])[:,None,None]

    # 创建一个与原始图像大小相同的空白图像
    segmented_image = np.copy(image)                            # (3,H,W)
    segmented_image = (segmented_image*std+mean)*255
    segmented_image = segmented_image.transpose((1, 2, 0))      # (H,W,3)

    # 遍历每个分割结果
    for class_id in range(1,5):
        mask = pred_mask==class_id
        # 获取类别对应的颜色
        color = color_mapping.get(class_id, (0, 0, 0))  # 默认为黑色
        # 使用颜色叠加到原图像上，以半透明的方式
        segmented_image[mask] = 0.5 * segmented_image[mask] + 0.5 * np.array(color, dtype=np.uint8)
    # 如果要保存叠加后的图像
    cv2.imwrite(save_dir, segmented_image)


def _plot_mask_(image, mask):
    # 创建颜色映射字典，将每个类别映射到不同的颜色
    color_mapping = {
        1: (0, 0, 255),  # 红色
        2: (0, 255, 0),  # 绿色
        3: (255, 0, 0),  # 蓝色
        4: (255, 255, 0)  # 黄色
    }
    for class_id in range(1,5):
        class_mask = mask==class_id
        # 获取类别对应的颜色
        color = color_mapping.get(class_id, (0, 0, 0))  # 默认为黑色
        # 使用颜色叠加到原图像上，以半透明的方式
        image[class_mask] = 0.5 * image[class_mask] + 0.5 * np.array(color, dtype=np.uint8)
    return image

def _plot_point_(image, points):
    color_list = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 粉红色
        (0, 255, 255)  # 青色
    ]
    for i,point in enumerate(points):
        cv2.circle(image, point.astype(np.int32), 5, color_list[i], -1)  # 绘制色点，半径为5
    return image


def plot_label_mask_point(data_dir=''):
    file_list = [line.strip().split() for line in open(data_dir)]
    files = read_files(file_list)

    for item in files:
        name = item["name"]
        image_path = item['img'].replace('..','data')
        keypoint_path = item['keypoint'].replace('..','data')
        mask_path = item['label'].replace('..','data')
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        mask = np.array(
            Image.open(mask_path).convert('P')
        )
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2])
        keypoint = np.round(keypoint)[:,:2]    # shape(6,3)

        # 绘制mask
        image = _plot_mask_(image, mask)
        # 绘制点
        image = _plot_point_(image, keypoint)

        save_dir = 'data/HipJoint/keypoint_mask_plot/'+name.replace('mask', 'label')+'.png'
        cv2.imwrite(save_dir,image)


def plot_pred_mask(data_dir=''):
    result_dict = torch.load(data_dir)
    save_dir = Path(data_dir).parent/'plot_mask_pred'
    save_dir.mkdir(parents=True, exist_ok=True)

    for name,item in result_dict.items():
        name
        image_path = 'data/HipJoint/images/'+name+'.jpg'
        mask_path = 'data/HipJoint/masks/'+name+'_mask.png'
        keypoint_path = 'data/HipJoint/annotations/'+name+'.json'
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        '''
        mask = np.array(
            Image.open(mask_path).convert('P')
        )
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2]) # shape(6,3)
        keypoint = np.round(keypoint)[:,:2]    # shape(6,2)
        '''
        mask_pred = item['pred_mask']
        keypoint_pred = item['pred_keypoint']
        keypoint_pred = keypoint_pred[:,[1,0]]

        # 绘制mask
        image = _plot_mask_(image, mask_pred)
        # 绘制点
        #image = _plot_point_(image, keypoint_pred)

        cv2.imwrite(str(save_dir/(name+'.png')),image)



def plot_pred_mask_point(data_dir=''):
    result_dict = torch.load(data_dir)
    save_dir = Path(data_dir).parent/'plot_mask_keypoint_pred'
    save_dir.mkdir(parents=True, exist_ok=True)

    for name,item in result_dict.items():
        name
        image_path = 'data/HipJoint/images/'+name+'.jpg'
        mask_path = 'data/HipJoint/masks/'+name+'_mask.png'
        keypoint_path = 'data/HipJoint/annotations/'+name+'.json'
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )   # shape(W,H,3)
        '''
        mask = np.array(
            Image.open(mask_path).convert('P')
        )
        with open(keypoint_path, "r") as json_file:
            keypoint = json.load(json_file)
        keypoint = [value+[int(id)] for id,value in keypoint.items() ]
        keypoint = sorted(keypoint, key=lambda x: x[2]) # shape(6,3)
        keypoint = np.round(keypoint)[:,:2]    # shape(6,2)
        '''
        mask_pred = item['pred_mask']
        keypoint_pred = item['pred_keypoint']
        keypoint_pred = keypoint_pred[:,[1,0]]

        # 绘制mask
        image = _plot_mask_(image, mask_pred)
        # 绘制点
        image = _plot_point_(image, keypoint_pred)

        cv2.imwrite(str(save_dir/(name+'.png')),image)



def plot_keypoint(image,keypoints,save_dir):
    '''
    # 读取图像
    image = cv2.imread('your_image.jpg')

    # 定义关键点坐标，这里使用示例坐标
    keypoints = [(100, 200), (300, 150), (400, 400)]
    '''
    color_list = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 粉红色
        (0, 255, 255)  # 青色
    ]

    # 遍历关键点并在图像上绘制
    for id,point in keypoints.items():
        id = int(id)    
        x, y = int(point[0]),int(point[1])
        # (x,y) = (W,H) , (横轴, 纵轴)
        cv2.circle(image, (x, y), 5, color_list[id-1], -1)  # 在图像上绘制红色圆圈

    # 保存标记后的图像
    cv2.imwrite(save_dir, image)


def delete_file():
    # 要删除文件的文件夹路径
    folder_path = 'output/HipJoint'

    # 定义文件名的匹配条件，例如，删除以 ".txt" 结尾的文件
    file_name_pattern = "keypoint_pred"

    # 获取文件夹中的文件列表
    file_list = os.listdir(folder_path)

    # 遍历文件列表并删除符合条件的文件
    for file_name in file_list:
        if file_name_pattern in file_name:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f"Deleted: {file_name}")

    print("Deletion completed.")


if __name__ == "__main__":
    '''
    root = 'data/HipJoint'
    anno_dir_list = (sorted(os.listdir(os.path.join(root, "annotations"))))
    images_dir_list = (sorted(os.listdir(os.path.join(root, "images"))))

    for i in range(len(anno_dir_list)):
        img_dir = images_dir_list[i]
        anno_dir = anno_dir_list[i]
        img_dir = os.path.join(root, "images",img_dir)
        anno_dir = os.path.join(root, "annotations",anno_dir)

        with open(anno_dir, "r") as json_file:
            anno_data = json.load(json_file)

        image = cv2.imread(img_dir)

        # 画图
        save_dir = img_dir.replace('images', 'keypoint_plot')
        plot_keypoint(image, anno_data, save_dir)
    '''

    plot_type = 0
    if plot_type == 0:
        plot_label_mask_point('data/HipJoint/fullset.txt')
    elif plot_type == 1:
        plot_pred_mask_point('output/hipjoint/seg_hrnet_w48_520_keypoint/predictions.pth')
    elif plot_type == 2:
        plot_pred_point('output/hipjoint/seg_hrnet_w48_520_keypoint_adam_LRstep copy 2/valid.pth', with_mask=True)
    elif plot_type == 3:
        plot_pred_mask('output/hipjoint/seg_hrnet_w48_cls59_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch200/predictions.pth')


    #pred_dir = 'output/HipJoint/face_alignment_cofw_hrnet_w18_5/predictions.pth'

    #plot_keypoint_label()
    #plot_keypoint_pred(pred_dir)


    #plot_line_label()

