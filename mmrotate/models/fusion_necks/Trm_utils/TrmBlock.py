# Author: Hu Yuxuan
# Date: 2022/8/30
# Modified: 2022/9/15
from torch import nn
from .MLP import Mlp
from .MHA import MultiheadAttention

class TrmBlock(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 num_heads, 
                 num_layers, 
                 mlp_ratio, 
                 no_layer_norm, 
                 no_out_layer_norm,
                 attn_drop,
                 mlp_drop):
        super(TrmBlock, self).__init__()
        self.num_layers = num_layers
        self.no_layer_norm = no_layer_norm
        self.no_out_layer_norm = no_out_layer_norm

        self.LayerNorm1 = nn.ModuleList()
        self.LayerNorm2 = nn.ModuleList()
            
        self.MHA = nn.ModuleList()
        self.mlp = nn.ModuleList()

        for _ in range(self.num_layers):
            if not self.no_layer_norm:
                self.LayerNorm1.append(nn.LayerNorm(emb_dim))
                self.LayerNorm2.append(nn.LayerNorm(emb_dim))
            else:
                self.LayerNorm1.append(nn.Identity())
                self.LayerNorm2.append(nn.Identity())
            self.MHA.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.mlp.append(Mlp(emb_dim, mlp_ratio, drop=mlp_drop))

        if (not self.no_layer_norm) and (not self.no_out_layer_norm):
            self.LayerNorm3 = nn.LayerNorm(emb_dim)
        else:
            self.LayerNorm3 = nn.Identity()

    def forward(self, x_q, x_k=None, x_v=None):
        x = x_q
        for i in range(self.num_layers):
            q = self.LayerNorm1[i](x)

            if x_k is None and x_v is None:
                k, v = q, q
            else:
                k = self.LayerNorm1[i](x_k)
                v = self.LayerNorm1[i](x_v)

            attn_output, attn_weights = self.MHA[i](q, k ,v)
            x1 = x + attn_output
            
            x2 = self.LayerNorm2[i](x1)
            x2 = self.mlp[i](x2)
            x = x2 + x1
        return self.LayerNorm3(x)