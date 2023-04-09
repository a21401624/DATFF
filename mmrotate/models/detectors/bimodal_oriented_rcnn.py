# Author: Hu Yuxuan
# Date: 2022/3/1
from ..builder import ROTATED_DETECTORS
from .bimodal_two_stage import BimodalRotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class BimodalOrientedRCNN(BimodalRotatedTwoStageDetector):

    def __init__(self,
                 backbone1,
                 backbone2,
                 fusion_neck,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(BimodalOrientedRCNN, self).__init__(
            backbone1=backbone1,
            backbone2=backbone2,
            fusion_neck=fusion_neck,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)