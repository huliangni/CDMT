_base_ = [
    '../models/swin_unet.py', '../datasets/hipjoint.py',
    '../default_runtime.py', '../schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(img_size=224),
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
)
    #test_cfg=dict(crop_size=(512, 512), stride=(170, 170)))