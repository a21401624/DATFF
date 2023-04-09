import torch
from torch import nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len, poem_type):
        super().__init__()
        self.d_model = d_model

        # 正余弦绝对位置编码，根据pos和i创建一个常量pe矩阵
        # 写法参考https://blog.csdn.net/Flying_sfeng/article/details/100996524
        if poem_type=='sinusoidal':
            pe = torch.zeros(max_seq_len, d_model)
            pos = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div_term)
            pe[:, 1::2] = torch.cos(pos * div_term)
            pe = pe.unsqueeze(1)
            self.register_buffer('pe', pe)
        # 可学习绝对位置编码
        elif poem_type=='learnable':
            self.pe = nn.Parameter(torch.randn(max_seq_len, 1, d_model))
        else:
            raise NotImplementedError("This type of positional embedding has not been implemented!")

    def forward(self, x):
        # 让embeddings vector 相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量
        seq_len = x.size(0)
        x = (x + self.pe[:seq_len])
        return x
