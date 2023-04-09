# Author: Hu Yuxuan
# Date: 2022/9/22
# Modified: 2023/3/30
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS

from ..Trm_utils import TrmBlock, PositionalEmbedding, ParTrmDecoder

@NECKS.register_module()
class DATFFFusionNeck(BaseModule):
    def __init__(self,
                 in_channels: list,
                 seq_length: list,
                 grids: list,
                 num_heads: list,
                 num_layers1: list,
                 num_layers2: list,
                 mlp_ratio: int,
                 proj_channel,
                 no_layer_norm,
                 no_out_layer_norm,
                 no_poem,
                 poem_drop,
                 attn_drop,
                 mlp_drop,
                 poem_type,
                 out_type,
                 **kwargs):
        super().__init__()
        grid_size = kwargs['grid_size']

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)

        if type(seq_length) == int:
            self.seq_length = [seq_length] * self.num_ins
        elif type(seq_length) == list:
            assert len(seq_length) == self.num_ins
            self.seq_length = seq_length

        if type(grids)==int:
            self.grids = [grids] * self.num_ins
        elif type(grids)==list:
            assert len(grids) == self.num_ins
            self.grids = grids

        if type(grid_size) == int:
            self.grid_size = [grid_size] * self.num_ins
        elif type(grid_size) == list:
            assert len(grid_size) == self.num_ins
            self.grid_size = grid_size

        chl_embed_dim = []
        for i in range(self.num_ins):
            chl_embed_dim.append(in_channels[i] // self.seq_length[i] * self.grid_size[i] * self.grid_size[i])

        if type(num_heads) == int:
            self.num_heads = [num_heads] * self.num_ins
        elif type(num_heads) == list:
            assert len(num_heads) == self.num_ins
            self.num_heads = num_heads
        
        if type(num_layers1) == int:
            self.num_layers1 = [num_layers1] * self.num_ins
        elif type(num_layers1) == list:
            assert len(num_layers1) == self.num_ins
            self.num_layers1 = num_layers1

        if type(num_layers2) == int:
            self.num_layers2 = [num_layers2] * self.num_ins
        elif type(num_layers2) == list:
            assert len(num_layers2)== self.num_ins
            self.num_layers2 = num_layers2

        if type(no_out_layer_norm) == bool:
            no_out_layer_norm = [no_out_layer_norm] * 2
        elif type(no_out_layer_norm) == list:
            assert len(no_out_layer_norm) == 2

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
            self.proj_fc1 = nn.ModuleList()
            self.proj_fc2 = nn.ModuleList()
            self.invproj_fc1 = nn.ModuleList()
            self.invproj_fc2 = nn.ModuleList()

        self.AvgPool1 = nn.ModuleList()
        self.CTrm_PoEms1 = nn.ModuleList()
        self.CTrm_PoEms2 = nn.ModuleList()
        if no_poem:
            assert poem_drop == 0
        self.poem_drop = nn.Dropout(poem_drop)
        self.CTrmBlocks1 = nn.ModuleList()
        self.CTrmBlocks2 = nn.ModuleList()
        self.FC1 = nn.ModuleList()
        self.FC2 = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        self.AvgPool2 = nn.ModuleList()
        self.PoEms1 = nn.ModuleList()
        self.PoEms2 = nn.ModuleList()
        self.ParTrmBlocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.identity = nn.Identity()
        if self.out_type == 'cat-conv':
            self.out = nn.ModuleList()
        for i in range(self.num_ins):
            self.AvgPool1.append(nn.AdaptiveAvgPool2d((self.grid_size[i], self.grid_size[i])))
            trm_block1 = TrmBlock(chl_embed_dim[i], 
                                 self.num_heads[i], 
                                 self.num_layers1[i], 
                                 self.mlp_ratio,
                                 no_layer_norm,
                                 no_out_layer_norm[0],
                                 attn_drop,
                                 mlp_drop)
            
            trm_block2 = TrmBlock(chl_embed_dim[i], 
                                 self.num_heads[i], 
                                 self.num_layers1[i], 
                                 self.mlp_ratio,
                                 no_layer_norm,
                                 no_out_layer_norm[0],
                                 attn_drop,
                                 mlp_drop)

            self.CTrm_PoEms1.append(PositionalEmbedding(chl_embed_dim[i], self.seq_length[i], poem_type))
            self.CTrm_PoEms2.append(PositionalEmbedding(chl_embed_dim[i], self.seq_length[i], poem_type))
            self.CTrmBlocks1.append(trm_block1)
            self.CTrmBlocks2.append(trm_block2)
            self.FC1.append(nn.Linear(self.grid_size[i] * self.grid_size[i], 1))
            self.FC2.append(nn.Linear(self.grid_size[i] * self.grid_size[i], 1))

            embed_dim = in_channels[i] if self.proj_channel is None else self.proj_channel[i]
            self.AvgPool2.append(nn.AdaptiveAvgPool2d((self.grids[i], self.grids[i])))
            if not no_poem:
                self.PoEms1.append(PositionalEmbedding(embed_dim, self.grids[i] * self.grids[i], poem_type))
                self.PoEms2.append(PositionalEmbedding(embed_dim, self.grids[i] * self.grids[i], poem_type))
            else:
                self.PoEms1.append(nn.Identity())
                self.PoEms2.append(nn.Identity())
            self.ParTrmBlocks.append(ParTrmDecoder(embed_dim, 
                                           self.num_heads[i], 
                                           self.num_layers2[i], 
                                           self.mlp_ratio,
                                           no_layer_norm,
                                           no_out_layer_norm[1],
                                           attn_drop,
                                           mlp_drop))

            if self.proj_channel is not None:
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

            self.upsample.append(nn.Upsample(size=(512//(2**(i+2)), 640//(2**(i+2))), mode='bilinear', align_corners=True))
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
        for i in range(len(inputs1)):
            F_T = inputs1[i]
            F_R = inputs2[i]
            
            s1 = self.AvgPool1[i](F_T)
            s2 = self.AvgPool1[i](F_R)
            s1 = self.identity(rearrange(s1, 'b (l c) h w -> l b (c h w)', l=self.seq_length[i]))
            s2 = self.identity(rearrange(s2, 'b (l c) h w -> l b (c h w)', l=self.seq_length[i]))
            s1 = self.poem_drop(self.CTrm_PoEms1[i](s1))
            s2 = self.poem_drop(self.CTrm_PoEms2[i](s2))
            Trm_output1 = self.CTrmBlocks1[i](s1, s2, s2)
            Trm_output2 = self.CTrmBlocks2[i](s2, s1, s1)

            Trm_output1 = rearrange(Trm_output1, 'l b (c h w) -> b (l c) (h w)', h=self.grid_size[i], w=self.grid_size[i])
            Trm_output2 = rearrange(Trm_output2, 'l b (c h w) -> b (l c) (h w)', h=self.grid_size[i], w=self.grid_size[i])
            
            Trm_output1 = torch.unsqueeze(self.FC1[i](Trm_output1), dim=-1)
            Trm_output2 = torch.unsqueeze(self.FC2[i](Trm_output2), dim=-1)

            Trm_output1 = self.sigmoid(Trm_output1)
            Trm_output2 = self.sigmoid(Trm_output2)

            F_T = F_T * Trm_output1
            F_R = F_R * Trm_output2
            
            h = F_T.shape[2]
            w = F_R.shape[3]
            input1_ = self.AvgPool2[i](F_T)
            input2_ = self.AvgPool2[i](F_R)
            if self.proj_channel is not None:
                input1_ = self.proj_fc1[i](input1_)
                input2_ = self.proj_fc2[i](input2_)
            input1_ = rearrange(input1_, 'b c h w -> (h w) b c')
            input2_ = rearrange(input2_, 'b c h w -> (h w) b c')

            Trm_input1 = self.poem_drop(self.PoEms1[i](input1_))
            Trm_input2 = self.poem_drop(self.PoEms2[i](input2_))

            Trm_output1, Trm_output2 = self.ParTrmBlocks[i](Trm_input1, Trm_input2)

            output1 = rearrange(Trm_output1, '(h w) b c -> b c h w', h=self.grids[i])
            output2 = rearrange(Trm_output2, '(h w) b c -> b c h w', h=self.grids[i])
            if self.proj_channel is not None:
                output1 = self.invproj_fc1[i](output1)
                output2 = self.invproj_fc2[i](output2)
            output1 = F.interpolate(output1, size=(h, w), mode='bilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(h, w), mode='bilinear', align_corners=True)
            fea_map1 = F_T + output1
            fea_map2 = F_R + output2
            if self.out_type == 'cat-conv':
                fused_fmap = self.out[i](torch.cat((fea_map1, fea_map2), dim=1))
            else:
                fused_fmap = fea_map1 + fea_map2
            outs.append(fused_fmap)
            
        return tuple(outs)