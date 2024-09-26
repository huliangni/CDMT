_base_ = ['../models/hrnet_w48.py','../datasets/hipjoint.py']

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