import numpy as np
import cv2
import os
import pickle
import json

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


'''
root = '../HipJoint'
pred_dir_list = (sorted(os.listdir(os.path.join(root, "pred"))))

for pred_dir in pred_dir_list:
    pred_dir = os.path.join(root, "pred",pred_dir)
    with open(pred_dir,'rb') as f:
        pred_mask = pickle.load(f)
    # 提取原图
    img_dir = pred_dir.replace('pred', 'images')
    img_dir = '_'.join(img_dir.split('_')[:-1])
    img_dir = img_dir+'.jpg'
    image = cv2.imread(img_dir)  # 替换为实际图像文件的路径
    # 保存mask
    save_dir = img_dir.replace('images', 'pred_plot')
    save_dir = save_dir.replace('jpg', 'png')
    plot_multi_mask(image, pred_mask, save_dir)
'''



if __name__ == "__main__":
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
