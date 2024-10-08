_base_ = [
    '../models/upernet_r50.py',
    '../datasets/hipjoint.py', '../default_runtime.py',
    '../schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5))