# Author: Hu Yuxuan
# Date: 2022/10/9
# Modified from /home1/huyuxuan/IGARSS/base_model/Trm_utils.py
import torch
from torch import nn
import torch.nn.functional as F
import math

def attention(q, k, v, d_k, mask=None, dropout=None):
    # q.shape:[bs,N,sl,d_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # mask掉那些为了padding长度增加的token，让其通过softmax计算后为0
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiheadAttention(nn.Module):
    """Custom MHA for the calculation of Parameter and FLOPs."""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.Dropout(dropout))


    def forward(self, q, k, v, mask=None):
        # q,k,v are of (L, B, D) shape. 
        # L: sequence length, B: batch size; D: embed dim.
        bs = q.size(1)

        # perform linear operation and split into N heads
        # get dimensions L * B * H * d_k
        k = self.k_linear(k).view(-1, bs, self.h, self.d_k)
        q = self.q_linear(q).view(-1, bs, self.h, self.d_k)
        v = self.v_linear(v).view(-1, bs, self.h, self.d_k)

        # transpose to get dimensions B * N * L * d_k
        k = k.transpose(0, 1).transpose(1, 2)
        q = q.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1).transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)  
        output = output.transpose(0, 1) # the same dimension as input k:(L, B, D)
        return output, scores