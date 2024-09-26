_base_ = ['../default_runtime.py']


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
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/whai362/PVT/'
            'releases/download/v2/pvt_v2_b2.pth'),
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


