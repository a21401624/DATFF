# Author: Hu Yuxuan
# Date: 2022/9/20
# Modified: 2023/3/22
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmcv.runner import BaseModule, auto_fp16
from ..Trm_utils import ParTrmDecoder, PositionalEmbedding
from mmdet.models.builder import NECKS

@NECKS.register_module()
class SCATFusionNeck(BaseModule):
    """Use ParTrmDecoder."""
    def __init__(self,
                 in_channels,
                 grids_h,
                 grids_w,
                 num_heads,
                 num_layers,
                 mlp_ratio,
                 proj_channel,
                 no_layer_norm,
                 no_out_layer_norm,
                 no_poem,
                 poem_drop,
                 attn_drop,
                 mlp_drop,
                 poem_type,
                 out_type):
        super(SCATFusionNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.num_ins = len(in_channels)
        self.in_channels = in_channels
        
        if type(grids_h)==int:
            self.grids_h = [grids_h] * self.num_ins
        elif type(grids_h)==list:
            assert len(grids_h) == self.num_ins
            self.grids_h = grids_h

        if type(grids_w)==int:
            self.grids_w = [grids_w] * self.num_ins
        elif type(grids_w)==list:
            assert len(grids_w) == self.num_ins
            self.grids_w = grids_w

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        assert out_type in ['add', 'cat-conv']
        self.out_type = out_type
        
        if type(proj_channel) == int:
            self.proj_channel = [proj_channel] * self.num_ins
        elif type(proj_channel) == list:
            assert len(proj_channel) == self.num_ins
            self.proj_channel = proj_channel
        else:
            self.proj_channel = proj_channel
        if self.proj_channel is not None:
            # self.proj_fc = nn.ModuleList()
            # self.invproj_fc = nn.ModuleList()
            self.proj_fc1 = nn.ModuleList()
            self.proj_fc2 = nn.ModuleList()
            self.invproj_fc1 = nn.ModuleList()
            self.invproj_fc2 = nn.ModuleList()

        self.AvgPool = nn.ModuleList()
        self.PoEms1 = nn.ModuleList()
        self.PoEms2 = nn.ModuleList()
        if no_poem:
            assert poem_drop == 0
        self.poem_drop = nn.Dropout(poem_drop)
        self.ParTrmBlocks = nn.ModuleList()
        if self.out_type == 'cat-conv':
            self.out = nn.ModuleList()
        for i in range(self.num_ins):
            embed_dim = in_channels[i] if self.proj_channel is None else self.proj_channel[i]
            self.AvgPool.append(nn.AdaptiveAvgPool2d((self.grids_h[i], self.grids_w[i])))
            if not no_poem:
                self.PoEms1.append(PositionalEmbedding(embed_dim, self.grids_h[i] * self.grids_w[i], poem_type))
                self.PoEms2.append(PositionalEmbedding(embed_dim, self.grids_h[i] * self.grids_w[i], poem_type))
            else:
                self.PoEms1.append(nn.Identity())
                self.PoEms2.append(nn.Identity())
            self.ParTrmBlocks.append(ParTrmDecoder(embed_dim, 
                                           self.num_heads, 
                                           self.num_layers, 
                                           self.mlp_ratio,
                                           no_layer_norm,
                                           no_out_layer_norm,
                                           attn_drop,
                                           mlp_drop))

            if self.proj_channel is not None:
                # self.proj_fc.append(nn.Linear(in_channels[i], self.proj_channel))
                # self.invproj_fc.append(nn.Linear(self.proj_channel, in_channels[i]))
                self.proj_fc1.append(nn.Conv2d(in_channels[i], self.proj_channel[i], 1))
                self.proj_fc2.append(nn.Conv2d(in_channels[i], self.proj_channel[i], 1))
                self.invproj_fc1.append(nn.Sequential(
                    nn.Conv2d(self.proj_channel[i], in_channels[i], 1),
                    nn.ReLU()
                    ))
                self.invproj_fc2.append(nn.Sequential(
                    nn.Conv2d(self.proj_channel[i], in_channels[i], 1),
                    nn.ReLU()
                    ))

            if self.out_type == 'cat-conv':
                f_conv = nn.Sequential(
                    nn.Conv2d(2 * in_channels[i], in_channels[i], 1),
                    nn.ReLU()
                )
                self.out.append(f_conv)


    def has_train_loss(self):
        return False

    @auto_fp16()
    def forward(self, inputs1, inputs2, **kwargs):
        """inputs1 and inputs2 are feature map lists, len(input1/input2)>=1"""
        assert len(inputs1) == len(self.in_channels)
        assert len(inputs2) == len(self.in_channels)

        outs = []
        for i in range(self.num_ins):
            input1 = inputs1[i]
            input2 = inputs2[i]
            h = input1.shape[2]
            w = input1.shape[3]
            input1_ = self.AvgPool[i](input1)
            input2_ = self.AvgPool[i](input2)
            if self.proj_channel is not None:
                input1_ = self.proj_fc1[i](input1_)
                input2_ = self.proj_fc2[i](input2_)
            input1_ = rearrange(input1_, 'b c h w -> (h w) b c')
            input2_ = rearrange(input2_, 'b c h w -> (h w) b c')

            Trm_input1 = self.poem_drop(self.PoEms1[i](input1_))
            Trm_input2 = self.poem_drop(self.PoEms2[i](input2_))

            Trm_output1, Trm_output2 = self.ParTrmBlocks[i](Trm_input1, Trm_input2)

            output1 = rearrange(Trm_output1, '(h w) b c -> b c h w', h=self.grids_h[i])
            output2 = rearrange(Trm_output2, '(h w) b c -> b c h w', h=self.grids_h[i])
            if self.proj_channel is not None:
                output1 = self.invproj_fc1[i](output1)
                output2 = self.invproj_fc2[i](output2)
            output1 = F.interpolate(output1, size=(h, w), mode='bilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(h, w), mode='bilinear', align_corners=True)
            fea_map1 = input1 + output1
            fea_map2 = input2 + output2
            if self.out_type == 'cat-conv':
                fused_fmap = self.out[i](torch.cat((fea_map1, fea_map2), dim=1))
            else:
                fused_fmap = fea_map1 + fea_map2
            outs.append(fused_fmap)

        return tuple(outs)
