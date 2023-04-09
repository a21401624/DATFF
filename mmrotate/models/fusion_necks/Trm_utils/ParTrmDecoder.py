# Author: Hu Yuxuan
# Date: 2022/9/20
# Modified: 2022/10/3
from torch import nn
from .MLP import Mlp
from .MHA import MultiheadAttention

class ParTrmDecoder(nn.Module):
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
        super(ParTrmDecoder, self).__init__()
        self.num_layers = num_layers
        self.no_layer_norm = no_layer_norm
        self.no_out_layer_norm = no_out_layer_norm

        self.LayerNorm1_1 = nn.ModuleList()
        self.LayerNorm1_2 = nn.ModuleList()
        self.LayerNorm1_3 = nn.ModuleList()
        self.LayerNorm2_1 = nn.ModuleList()
        self.LayerNorm2_2 = nn.ModuleList()
        self.LayerNorm2_3 = nn.ModuleList()
            
        self.MHSA1 = nn.ModuleList()
        self.MHCA1 = nn.ModuleList()
        self.MLP1 = nn.ModuleList()

        self.MHSA2 = nn.ModuleList()
        self.MHCA2 = nn.ModuleList()
        self.MLP2 = nn.ModuleList()

        for _ in range(self.num_layers):
            if not self.no_layer_norm:
                self.LayerNorm1_1.append(nn.LayerNorm(emb_dim))
                self.LayerNorm1_2.append(nn.LayerNorm(emb_dim))
                self.LayerNorm1_3.append(nn.LayerNorm(emb_dim))
                self.LayerNorm2_1.append(nn.LayerNorm(emb_dim))
                self.LayerNorm2_2.append(nn.LayerNorm(emb_dim))
                self.LayerNorm2_3.append(nn.LayerNorm(emb_dim))
            else:
                self.LayerNorm1_1.append(nn.Identity())
                self.LayerNorm1_2.append(nn.Identity())
                self.LayerNorm1_3.append(nn.Identity())
                self.LayerNorm2_1.append(nn.Identity())
                self.LayerNorm2_2.append(nn.Identity())
                self.LayerNorm2_3.append(nn.Identity())
            self.MHSA1.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.MHCA1.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.MLP1.append(Mlp(emb_dim, mlp_ratio, drop=mlp_drop))

            self.MHSA2.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.MHCA2.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.MLP2.append(Mlp(emb_dim, mlp_ratio, drop=mlp_drop))

        if (not self.no_layer_norm) and (not self.no_out_layer_norm):
            self.LayerNorm1_4 = nn.LayerNorm(emb_dim)
            self.LayerNorm2_4 = nn.LayerNorm(emb_dim)
        else:
            self.LayerNorm1_4 = nn.Identity()
            self.LayerNorm2_4 = nn.Identity()

    def forward(self, input1, input2):
        x = input1
        y = input2
        for i in range(self.num_layers):              
            q1 = self.LayerNorm1_1[i](x)
            q2 = self.LayerNorm2_1[i](y)

            attn_output1, attn_weights1 = self.MHSA1[i](q1, q1 ,q1)
            x1 = x + attn_output1
            
            attn_output2, attn_weights2 = self.MHSA2[i](q2, q2 ,q2)
            y1 = y + attn_output2

            # One MHA corresponds to one LayerNorm?
            q1 = self.LayerNorm1_2[i](x1)
            # k1 = self.LayerNorm1_2[i](y1)
            # v1 = k1

            q2 = self.LayerNorm2_2[i](y1)
            # k2 = self.LayerNorm2_2[i](x1)
            # v2 = k2

            attn_output3, attn_weights3 = self.MHCA1[i](q1, q2, q2)
            x2 = x1 + attn_output3

            attn_output4, attn_weights4 = self.MHCA2[i](q2, q1, q1)
            y2 = y1 + attn_output4

            x3 = self.LayerNorm1_3[i](x2)
            x3 = self.MLP1[i](x3)
            x = x3 + x2
            
            y3 = self.LayerNorm2_3[i](y2)
            y3 = self.MLP2[i](y3)
            y = y3 + y2
        return self.LayerNorm1_4(x), self.LayerNorm2_4(y)