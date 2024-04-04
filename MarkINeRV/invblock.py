from math import exp
import torch
import torch.nn as nn
import config as c
from rrdb_denselayer import ResidualDenseBlock_out


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True, in_1=3, in_2=3):#clamp将一个值或一系列值限制在两个极端之间
        super().__init__()#__ init__ ()方法必须包含一个self参数，而且要是第一个参数。
        if harr:
            self.split_len1 = in_1 * 4#小波变换对图像进行分䣓 harr特征提取代码小波分解
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ函数
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η函数
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ函数
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))#e,其中包含一个激活函数

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),#torch.narrow(input, dim, start, length) → Tensor
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


