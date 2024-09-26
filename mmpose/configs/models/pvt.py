_base_ = ['../default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='PyramidVisionTransformer',
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/huliangni/Segmentation/mmpose/storage/pvt_small.pth'),
    ),
    neck=dict(type='FeatureMapProcessor', select_index=3),
    head=dict(
        type='HeatmapHead',
        in_channels=512,
        out_channels=17,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

