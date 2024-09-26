from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class HipJointDataset(BaseSegDataset):

    METAINFO = dict(
        classes = ('background', 'region1', 'region2', 'region3', 'region4'), 
        palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123]])
    
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)