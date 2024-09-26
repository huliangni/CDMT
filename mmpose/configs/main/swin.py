_base_ = ['../models/swin.py','../datasets/hipjoint.py']

# codec settings
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=1.5)

model = dict(
    head = dict(
        decoder=codec,
        out_channels=6
    ),
    test_cfg = dict(output_heatmaps=True)
)

# runtime
train_cfg = dict(max_epochs=120, val_interval=1)

# hooks
default_hooks = dict(checkpoint=dict(save_best='NME', rule='less', interval=220))
