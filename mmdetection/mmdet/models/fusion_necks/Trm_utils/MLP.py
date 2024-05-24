# Author: Hu Yuxuan
# Date: 2022/8/30
from torch import nn

class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio, act='GELU', drop=0.):
        super().__init__()
        assert act in ['ReLU', 'GELU']
        out_features = in_features
        hidden_features = in_features * mlp_ratio
        self.fc1 = nn.Linear(in_features, hidden_features)
        if act == 'ReLU':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x