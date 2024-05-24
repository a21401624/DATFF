# The CMAFF Fusion Module is an implementation based on the paper 
# 'Cross-Modality Attentive Feature Fusion for Object Detection in Multispectral Remote Sensing Imagery'.
# Author: Hu Yuxuan
# Date: 2022/5/6
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS

@NECKS.register_module()
class CMAFFFusionNeck(BaseModule):

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CMAFFFusionNeck, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        self.act_cfg = act_cfg

        # Differential Enhancive Module
        self.shared_convs = nn.ModuleList()
        # Common Selective Module
        # We use conv layers to perform the same function of FC layers.
        self.shared_fc1 = nn.ModuleList()
        self.fc2_R = nn.ModuleList()
        self.fc2_T = nn.ModuleList()
        self.out = nn.ModuleList()
        for i in range(self.num_ins):
            shared_conv = ConvModule(
                in_channels[i],
                in_channels[i],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            shared_fc1 = ConvModule(
                in_channels[i],
                in_channels[i]//32,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fc2_R = ConvModule(
                in_channels[i]//32,
                in_channels[i],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fc2_T = ConvModule(
                in_channels[i]//32,
                in_channels[i],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            
            self.shared_convs.append(shared_conv)
            self.shared_fc1.append(shared_fc1)
            self.fc2_R.append(fc2_R)
            self.fc2_T.append(fc2_T)
            self.out.append(nn.Identity())

    def has_train_loss(self):
        return False

    @auto_fp16()
    def forward(self, inputs1, inputs2, **kwargs):
        """inputs1 and inputs2 are feature map lists, len(input1/input2)>=1"""
        assert len(inputs1) == len(self.in_channels)
        assert len(inputs2) == len(self.in_channels)

        outs = []
        for i in range(len(inputs1)):
            F_R = inputs2[i]
            F_T = inputs1[i]
            # Differential Enhancive Module
            F_D = F_R - F_T
            s1 = F.adaptive_avg_pool2d(F_D, (1,1))
            s2 = F.adaptive_max_pool2d(F_D, (1,1))
            z1 = self.shared_convs[i](s1)
            z2 = self.shared_convs[i](s2)
            M_DM = torch.sigmoid((z1+z2))
            F_DM = F_R * (1 + M_DM) + F_T * (1 + M_DM)
            # Common Selective Module
            F_C = F_R + F_T
            s = F.adaptive_avg_pool2d(F_C, (1,1))
            z1 = self.fc2_R[i](self.shared_fc1[i](s))
            z2 = self.fc2_T[i](self.shared_fc1[i](s))
            bs = z1.shape[0]
            chl = z1.shape[1]
            z1 = torch.reshape(z1, (bs, 1, chl))
            z2 = torch.reshape(z2, (bs, 1, chl))
            z = torch.cat([z1, z2], dim=1)
            z = torch.softmax(z, dim=1)
            M_R_CM = torch.reshape(z[:, 0, :], (bs, chl, 1, 1)) 
            M_T_CM = torch.reshape(z[:, 1, :], (bs, chl, 1, 1)) 
            F_CM = F_R * M_R_CM + F_T * M_T_CM
            F_FUSE = F_DM + F_CM
            outs.append(self.out[i](F_FUSE))
            
        return tuple(outs)