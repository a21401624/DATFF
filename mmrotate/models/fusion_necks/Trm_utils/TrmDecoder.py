# Author: Hu Yuxuan
# Date: 2022/10/4
from torch import nn
from .MLP import Mlp

class TrmDecoder(nn.Module):
    """Both multihead self-attention and multihead cross-attention is implemented."""
    def __init__(self, 
                 emb_dim, 
                 num_heads, 
                 num_layers, 
                 mlp_ratio, 
                 no_layer_norm, 
                 no_out_layer_norm,
                 attn_drop,
                 mlp_drop):
        super(TrmDecoder, self).__init__()
        self.num_layers = num_layers
        self.no_layer_norm = no_layer_norm
        self.no_out_layer_norm = no_out_layer_norm

        self.LayerNorm1 = nn.ModuleList()
        self.LayerNorm2 = nn.ModuleList()
        self.LayerNorm3 = nn.ModuleList()
            
        self.MHSA = nn.ModuleList()
        self.MHCA = nn.ModuleList()
        self.MLP = nn.ModuleList()
        for _ in range(self.num_layers):
            if not self.no_layer_norm:
                self.LayerNorm1.append(nn.LayerNorm(emb_dim))
                self.LayerNorm2.append(nn.LayerNorm(emb_dim))
                self.LayerNorm3.append(nn.LayerNorm(emb_dim))
            else:
                self.LayerNorm1.append(nn.Identity())
                self.LayerNorm2.append(nn.Identity())
                self.LayerNorm3.append(nn.Identity())
            self.MHSA.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.MHCA.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.MLP.append(Mlp(emb_dim, mlp_ratio, drop=mlp_drop))

        if (not self.no_layer_norm) and (not self.no_out_layer_norm):
            self.LayerNorm4 = nn.LayerNorm(emb_dim)
        else:
            self.LayerNorm4 = nn.Identity()

    def forward(self, input1, input2):
        x = input1
        y = input2
        for i in range(self.num_layers):              
            q1 = self.LayerNorm1[i](x)

            attn_output1, attn_weights1 = self.MHSA[i](q1, q1 ,q1)
            x1 = x + attn_output1

            # One MHA corresponds to one LayerNorm?
            q1 = self.LayerNorm2[i](x1)
            k1 = self.LayerNorm2[i](y)
            v1 = k1

            attn_output2, attn_weights2 = self.MHCA[i](q1, k1 ,v1)
            x2 = x1 + attn_output2

            x3 = self.LayerNorm3[i](x2)
            x3 = self.MLP[i](x3)
            x = x3 + x2
        return self.LayerNorm4(x)