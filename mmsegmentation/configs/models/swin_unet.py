# model settings
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinUnet',
        img_size=224,
        patch_size=4, 
        in_chans=3, 
        num_classes=1000,
        embed_dim=96, 
        depths=[2, 2, 2, 2], 
        depths_decoder=[2, 2, 2, 1], 
        num_heads=[3, 6, 12, 24],
        window_size=7, 
        mlp_ratio=4., 
        qkv_bias=True, 
        qk_scale=None,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.1,
        ape=False, 
        patch_norm=True,
        use_checkpoint=False, 
        final_upsample="expand_first", 
        ),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=256, stride=170))
