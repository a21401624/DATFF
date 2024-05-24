# Author: Hu Yuxuan
# Date: 2022/9/13
import torch
import torch.nn as nn
from einops import rearrange

from mmcv.runner import BaseModule, auto_fp16

from .Trm_utils import TrmBlock, PositionalEmbedding
from mmdet.models.builder import NECKS

@NECKS.register_module()
class CCATFusionNeck(BaseModule):

    def __init__(self,
                 in_channels: list,
                 seq_length: list,
                 num_heads: list,
                 num_layers: list,
                 mlp_ratio: int,
                 no_layer_norm,
                 no_out_layer_norm,
                 poem_drop,
                 attn_drop,
                 mlp_drop,
                 poem_type,
                 out_type,
                 **kwargs):
        super().__init__()
        assert isinstance(in_channels, list)
        grid_size = kwargs['grid_size']
        self.in_channels = in_channels
        self.num_ins = len(in_channels)

        if type(seq_length) == int:
            self.seq_length = [seq_length] * self.num_ins
        elif type(seq_length) == list:
            assert len(seq_length) == self.num_ins
            self.seq_length = seq_length

        if type(num_heads) == int:
            self.num_heads = [num_heads] * self.num_ins
        elif type(num_heads) == list:
            assert len(num_heads) == self.num_ins
            self.num_heads = num_heads
        
        if type(num_layers) == int:
            self.num_layers = [num_layers] * self.num_ins
        elif type(num_layers) == list:
            assert len(num_layers) == self.num_ins
            self.num_layers = num_layers

        self.mlp_ratio = mlp_ratio
        assert out_type in ['add', 'cat-conv']
        self.out_type = out_type
        
        if type(grid_size) == int:
            self.grid_size = [grid_size] * self.num_ins
        elif type(grid_size) == list:
            assert len(grid_size) == self.num_ins
            self.grid_size = grid_size

        embed_dim = []
        for i in range(self.num_ins):
            embed_dim.append((in_channels[i] // self.seq_length[i]) * self.grid_size[i] * self.grid_size[i])
        
        self.AvgPool = nn.ModuleList()
        self.PoEms1 = nn.ModuleList()
        self.PoEms2 = nn.ModuleList()
        self.poem_drop = nn.Dropout(poem_drop)
        self.TrmBlocks1 = nn.ModuleList()
        self.TrmBlocks2 = nn.ModuleList()
        self.FC1 = nn.ModuleList()
        self.FC2 = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self.identity = nn.Identity() # For visualize feature maps.
        self.out = nn.ModuleList()
        for i in range(self.num_ins):
            self.AvgPool.append(nn.AdaptiveAvgPool2d((self.grid_size[i], self.grid_size[i])))
            trm_block1 = TrmBlock(embed_dim[i], 
                                 self.num_heads[i], 
                                 self.num_layers[i], 
                                 self.mlp_ratio,
                                 no_layer_norm,
                                 no_out_layer_norm,
                                 attn_drop,
                                 mlp_drop)
            
            trm_block2 = TrmBlock(embed_dim[i], 
                                 self.num_heads[i], 
                                 self.num_layers[i], 
                                 self.mlp_ratio,
                                 no_layer_norm,
                                 no_out_layer_norm,
                                 attn_drop,
                                 mlp_drop)
            
            f_conv = nn.Sequential(
                nn.Conv2d(2 * in_channels[i], in_channels[i], 1),
                nn.ReLU()
            )

            self.PoEms1.append(PositionalEmbedding(embed_dim[i], self.seq_length[i], poem_type))
            self.PoEms2.append(PositionalEmbedding(embed_dim[i], self.seq_length[i], poem_type))
            self.TrmBlocks1.append(trm_block1)
            self.TrmBlocks2.append(trm_block2)
            self.FC1.append(nn.Linear(self.grid_size[i] * self.grid_size[i], 1))
            self.FC2.append(nn.Linear(self.grid_size[i] * self.grid_size[i], 1))
            self.out.append(f_conv)

    def has_train_loss(self):
        return False

    @auto_fp16()
    def forward(self, inputs1, inputs2, **kwargs):
        """inputs1 and inputs2 are feature map lists, len(input1/input2)>=1"""
        assert len(inputs1) == len(self.in_channels)
        assert len(inputs2) == len(self.in_channels)

        outs = []
        for i in range(len(inputs1)):
            F_T = inputs1[i]
            F_R = inputs2[i]
            s1 = self.AvgPool[i](F_T)
            s2 = self.AvgPool[i](F_R)

            s1 = rearrange(s1, 'b (l c) h w -> l b (c h w)', l=self.seq_length[i])
            s2 = rearrange(s2, 'b (l c) h w -> l b (c h w)', l=self.seq_length[i])
            s1 = self.poem_drop(self.PoEms1[i](s1))
            s2 = self.poem_drop(self.PoEms2[i](s2))
            Trm_output1 = self.TrmBlocks1[i](s1, s2, s2)
            Trm_output2 = self.TrmBlocks2[i](s2, s1, s1)

            Trm_output1 = rearrange(Trm_output1, 'l b (c h w) -> b (l c) (h w)', h=self.grid_size[i], w=self.grid_size[i])
            Trm_output2 = rearrange(Trm_output2, 'l b (c h w) -> b (l c) (h w)', h=self.grid_size[i], w=self.grid_size[i])

            Trm_output1 = torch.unsqueeze(self.FC1[i](Trm_output1), dim=-1)
            Trm_output2 = torch.unsqueeze(self.FC2[i](Trm_output2), dim=-1)

            Trm_output1 = self.identity(self.sigmoid(Trm_output1))
            Trm_output2 = self.identity(self.sigmoid(Trm_output2))

            F_T = F_T * Trm_output1
            F_R = F_R * Trm_output2
            if self.out_type == 'cat-conv':
                fused_fmap = self.out[i](torch.cat((F_T, F_R), dim=1))
            else:
                fused_fmap = F_T + F_R
            outs.append(fused_fmap)
            
        return tuple(outs)