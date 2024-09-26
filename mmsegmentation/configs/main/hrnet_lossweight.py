_base_ = [
    '../models/fcn_hr18.py', '../datasets/hipjoint.py',
    '../default_runtime.py', '../schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    data_preprocessor=data_preprocessor, 
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(
        num_classes=5,
        loss_decode=dict(class_weight=[0.2,1,1,1,1]),
        norm_cfg=norm_cfg
    )
)
