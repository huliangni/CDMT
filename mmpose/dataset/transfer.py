import json
import os
import numpy as np
from PIL import Image

def load_COCO(file_dir):
    #root_dir = '/home/HDD/hln/dataset/300W/'
    #file_name = 'face_landmarks_300w_test.json'
    #root_dir = '/home/HDD/hln/dataset/COCO2017/COCO2017/raw/Annotations/'
    #file_name = 'person_keypoints_val2017.json'
    #ile_dir = root_dir+'annotations/'+file_name
    # 打开JSON文件并读取数据
    with open(file_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_hipjoint():
    """
    return {
        'image_name':[],
        'annotations':[(x,y)],
        'mask':[]    shape(y,x) = (height,width)
    }
    """
    # 获取源路径和文件
    origin_root = '/home/hln/dataset/HipJoint/'
    point_dir_list = os.listdir(origin_root+'annotations_calibration')
    img_dir_list = [name[:-5]+'.jpg' for name in point_dir_list ]
    mask_dir_list = [name[:-5]+'_mask'+'.png' for name in point_dir_list ]

    # 读取源文件
    origin_info = {
        'image_name':[],
        'annotations':[],
        'mask':[]
    }

    for i in range(len(point_dir_list)):
        point_dir = origin_root+'annotations_calibration/'+point_dir_list[i]
        with open(point_dir, 'r', encoding='utf-8') as file:
            annotation = json.load(file)
        mask_dir = origin_root+'masks/'+mask_dir_list[i]
        mask = Image.open(mask_dir)
        mask = np.array(mask)
        origin_info['image_name'].append(img_dir_list[i])
        origin_info['annotations'].append(annotation)
        origin_info['mask'].append(mask)

    return origin_info


def hipjoint2COCO(origin_info, coco_output_file):
    # COCO格式需要的类别信息，这里假设只有一个类别
    categories = [{
        "id": 1,
        "name": "hipjoint",
        "supercategory": "hipjoint",
        "keypoints": [],  # 根据实际情况填写
        "skeleton": []  # 根据实际情况填写
    }]

    # 存储COCO格式的JSON数据
    coco_data = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    # 读取原始文件并转换
    for id in range(len(origin_info['image_name'])):    
        # 创建COCO格式的images条目
        image_info = {
            "id": id,  # 假设id从0开始递增
            "file_name": origin_info['image_name'][id],
            'height':origin_info['mask'][id].shape[0],
            'width':origin_info['mask'][id].shape[1]
        }
        coco_data["images"].append(image_info)
        
        keypoints = []
        for k in range(1,1+6):
            keypoints += origin_info['annotations'][id][str(k)]
            keypoints.append(1)
        
        mask = origin_info['mask'][id]
        index = np.where(mask>0)    # (y_list,x_list)
        y_min = index[0].min().item()
        y_max = index[0].max().item()
        x_min = index[1].min().item()
        x_max = index[1].max().item()
        height = y_max-y_min
        width = x_max-x_min
        
        # 创建COCO格式的annotations条目
        annotation_info = {
            "id": id,
            "image_id": id,  # 引用刚添加的图像id
            "category_id": 1,  # 假设只有一个类别
            "keypoints": keypoints,  # 存储转换后的关键点
            "num_keypoints": 6,  # 初始化关键点数量
            "bbox":[x_min,y_min,width,height],
            "area":int(width*height),
            "iscrowd":0
        }
        
        coco_data["annotations"].append(annotation_info)

    # 将转换后的数据写入JSON文件
    with open(coco_output_file, 'w') as outfile:
        json.dump(coco_data, outfile, indent=4)

    print(f'COCO格式的JSON文件已生成在：{coco_output_file}')


def split_dataset(output_root,split_type):

    split_dir = output_root+f'splits/{split_type}.txt'
    output_save_dir = output_root+f'annotations/hipjoint_{split_type}.json'

    data = load_COCO(output_root+'annotations/hipjoint_all.json')

    with open(split_dir, 'r', encoding='utf-8') as file:
        split_info = [line.strip() for line in file]

    split_images = []
    split_annotations = []

    image_list = [item['file_name'][:-4] for item in data['images'] ]

    for id,name in enumerate(image_list):
        if name in split_info:
            split_images.append(data['images'][id])
            split_annotations.append(data['annotations'][id])

    split_data = {
        'info':data['info'],
        'licenses':data['licenses'],
        'categories':data['categories'],
        'images':split_images,
        'annotations':split_annotations
    }

    # 将转换后的数据写入JSON文件
    with open(output_save_dir, 'w') as outfile:
        json.dump(split_data, outfile, indent=4)

    print(f'COCO格式的JSON文件已生成在：{output_save_dir}')


output_root = '/home/hln/dataset/HipJoint_mmseg/'

# 加载hipjoint数据
#origin_info = load_hipjoint()

# 处理成coco格式
#hipjoint2COCO(origin_info,coco_output_file = output_root+'annotations/hipjoint_all.json')

# 加载处理后的coco数据
#data = load_COCO(coco_output_file)

# 分割数据集为train, valid, test
split_dataset(output_root,'train')
split_dataset(output_root,'valid')
split_dataset(output_root,'test')


print(0)