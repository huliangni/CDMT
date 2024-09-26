
* 处理步骤
  * 把dataset的hipjoint_dataset.py文件复制到mmpose/datasets/datasets，并修改里面的__init__.py
  

**目录说明**
* configs
    * 存放config，从mmpose github上找，然后修改
* dataset
    * hipjoint_dataset.py: 是要放在mmpose/datasets的数据集文件
    * transfer.py: 转化数据集为coco格式
* tools
    * train.py 训练文件
    * test.py 测试文件
    * evaluate.ipynb: 评估结果，计算指标
* scrips