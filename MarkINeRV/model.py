import torch.optim
import torch.nn as nn
import config as c
from hinet import Hinet


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = Hinet()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:

            out = self.model(x, rev=True)

        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')#按.进行拆分，拆分后会形成一个字符串的数组并返回。
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()#返回一个符合均值为0，方差为1的正态分布（标准正态分布）中填充随机数的张量
            if split[-2] == 'conv5':
                param.data.fill_(0.)#用0填充
