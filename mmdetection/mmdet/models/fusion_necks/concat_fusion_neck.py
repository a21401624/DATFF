# Author: Hu Yuxuan
# Date: 2022/3/1
# Modified: 2022/4/28
#           2022/8/24
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS

@NECKS.register_module()
class ConcatFusionNeck(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 no_conv,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 version='0.0'):
        super(ConcatFusionNeck, self).__init__(init_cfg)
        assert version == '0.0'
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert len(in_channels) == len(out_channels)
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_conv = no_conv
        #TODO num_outs don't need to be the same with num_ins
        assert self.num_outs == self.num_ins

        if not self.no_conv:
            self.fusion_convs = nn.ModuleList()

            for i in range(self.num_ins):
                f_conv = ConvModule(
                    2 * in_channels[i],
                    out_channels[i],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                
                self.fusion_convs.append(f_conv)

    def has_train_loss(self):
        return False

    @auto_fp16()
    def forward(self, inputs1, inputs2, **kwargs):
        """inputs1 and inputs2 are feature map lists, len(input1/input2)>=1"""
        assert len(inputs1) == len(self.in_channels)
        assert len(inputs2) == len(self.in_channels)

        outs = []
        if self.no_conv:
            outs = [
                torch.cat((inputs1[i], inputs2[i]), dim=1)# + inputs1[i] + inputs2[i]
                        for i in range(self.num_outs)
            ]
        else:
            outs = [
                self.fusion_convs[i](torch.cat((inputs1[i], inputs2[i]), dim=1))# + inputs1[i] + inputs2[i]
                        for i in range(self.num_outs)
            ]
        return tuple(outs)