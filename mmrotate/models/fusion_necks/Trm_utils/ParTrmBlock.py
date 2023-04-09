# Author: Hu Yuxuan
# Date: 2022/9/19
from torch import nn
from .MLP import Mlp

class ParTrmBlock(nn.Module):
    """Parallel Multilayer Cross Transformers."""
    def __init__(self, 
                 emb_dim, 
                 num_heads, 
                 num_layers, 
                 mlp_ratio, 
                 no_layer_norm, 
                 no_out_layer_norm,
                 attn_drop,
                 mlp_drop):
        super(ParTrmBlock, self).__init__()
        self.num_layers = num_layers
        self.no_layer_norm = no_layer_norm
        self.no_out_layer_norm = no_out_layer_norm

        self.LayerNorm1_1 = nn.ModuleList()
        self.LayerNorm1_2 = nn.ModuleList()
        self.LayerNorm2_1 = nn.ModuleList()
        self.LayerNorm2_2 = nn.ModuleList()
            
        self.MHA1 = nn.ModuleList()
        self.MLP1 = nn.ModuleList()

        self.MHA2 = nn.ModuleList()
        self.MLP2 = nn.ModuleList()

        for _ in range(self.num_layers):
            if not self.no_layer_norm:
                self.LayerNorm1_1.append(nn.LayerNorm(emb_dim))
                self.LayerNorm1_2.append(nn.LayerNorm(emb_dim))
                self.LayerNorm2_1.append(nn.LayerNorm(emb_dim))
                self.LayerNorm2_2.append(nn.LayerNorm(emb_dim))
            else:
                self.LayerNorm1_1.append(nn.Identity())
                self.LayerNorm1_2.append(nn.Identity())
                self.LayerNorm2_1.append(nn.Identity())
                self.LayerNorm2_2.append(nn.Identity())
            self.MHA1.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.MLP1.append(Mlp(emb_dim, mlp_ratio, drop=mlp_drop))

            self.MHA2.append(nn.MultiheadAttention(emb_dim, num_heads, attn_drop))
            self.MLP2.append(Mlp(emb_dim, mlp_ratio, drop=mlp_drop))

        if (not self.no_layer_norm) and (not self.no_out_layer_norm):
            self.LayerNorm1_3 = nn.LayerNorm(emb_dim)
            self.LayerNorm2_3 = nn.LayerNorm(emb_dim)
        else:
            self.LayerNorm1_3 = nn.Identity()
            self.LayerNorm2_3 = nn.Identity()

    def forward(self, input1, input2):
        x = input1
        y = input2
        for i in range(self.num_layers): 
            # One MHA corresponds to one LayerNorm?             
            q1 = self.LayerNorm1_1[i](x)
            k1 = self.LayerNorm1_1[i](y)
            v1 = k1

            q2 = self.LayerNorm2_1[i](y)
            k2 = self.LayerNorm2_1[i](x)
            v2 = k2

            attn_output1, attn_weights1 = self.MHA1[i](q1, k1 ,v1)
            x1 = x + attn_output1

            attn_output2, attn_weights2 = self.MHA2[i](q2, k2 ,v2)
            y1 = y + attn_output2

            x2 = self.LayerNorm1_2[i](x1)
            x2 = self.MLP1[i](x2)
            x = x2 + x1
            
            y2 = self.LayerNorm2_2[i](y1)
            y2 = self.MLP2[i](y2)
            y = y2 + y1
        return self.LayerNorm1_3(x), self.LayerNorm2_3(y)