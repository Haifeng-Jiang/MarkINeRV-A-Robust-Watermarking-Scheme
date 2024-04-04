import torch
import torch.nn as nn
import modules.module_util as mutil


# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)#in_channels=input=四维张量[N, C, H, W]中的C,out_channels=32=3*4,kernel_size=3,stride = 1,padding=1
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        # self.conv6 = nn.Conv2d(input + 5 * 32, 32, 3, 1, 1, bias=bias)
        # self.conv7 = nn.Conv2d(input + 6 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        mutil.initialize_weights([self.conv5], 0.)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.lrelu(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        # # x6 = self.lrelu(nn.BatchNorm2d(int(x5+x)))

        # x6 = self.lrelu(self.conv6(torch.cat((x, x1, x2, x3, x4 , x5), 1)))#x5+x#改进的RDN从密集改为残差密集
        # x7 = self.lrelu(self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1)))
        return x5
# import BasicModule
# import torch.nn as nn
# import torch
# import time
# class one_conv(nn.Module):
#     def __init__(self,inchanels,growth_rate,kernel_size = 3):
#         super(one_conv,self).__init__()
#         self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
#         self.relu = nn.ReLU()
#     def forward(self,x):
#         output = self.relu(self.conv(x))
#         return torch.cat((x,output),1)
#
# class RDB(nn.Module):
#     def __init__(self,G0,C,G,kernel_size = 3):
#         super(RDB,self).__init__()
#         convs = []
#         for i in range(C):
#             convs.append(one_conv(G0+i*G,G))
#         self.conv = nn.Sequential(*convs)
#         #local_feature_fusion
#         self.LFF = nn.Conv2d(G0+C*G,G0,kernel_size = 1,padding = 0,stride =1)
#     def forward(self,x):
#         out = self.conv(x)
#         lff = self.LFF(out)
#         #local residual learning
#         return lff + x
#
# class rdn(BasicModule.basic):
#     def __init__(self,opts):
#         '''
#         opts: the system para
#         '''
#         super(rdn,self).__init__()
#         '''
#         D: RDB number 20
#         C: the number of conv layer in RDB 6
#         G: the growth rate 32
#         G0:local and global feature fusion layers 64filter
#         '''
#         self.D = opts.D
#         self.C = opts.C
#         self.G = opts.G
#         self.G0 = opts.G0
#         print "D:{},C:{},G:{},G0:{}".format(self.D,self.C,self.G,self.G0)
#         kernel_size =opts.kernel_size
#         input_channels = opts.input_channels
#         #shallow feature extraction
#         self.SFE1 = nn.Conv2d(input_channels,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride=  1)
#         self.SFE2 = nn.Conv2d(self.G0,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride =1)
#         #RDB for paper we have D RDB block
#         self.RDBS = nn.ModuleList()
#         for d in range(self.D):
#             self.RDBS.append(RDB(self.G0,self.C,self.G,kernel_size))
#         #Global feature fusion
#         self.GFF = nn.Sequential(
#                nn.Conv2d(self.D*self.G0,self.G0,kernel_size = 1,padding = 0 ,stride= 1),
#                nn.Conv2d(self.G0,self.G0,kernel_size,padding = kernel_size>>1,stride = 1),
#         )
#         #upsample net
#         self.up_net = nn.Sequential(
#                 nn.Conv2d(self.G0,self.G*4,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1),
#                 nn.PixelShuffle(2),
#                 nn.Conv2d(self.G,self.G*4,kernel_size = kernel_size,padding =kernel_size>>1,stride = 1),
#                 nn.PixelShuffle(2),
#                 nn.Conv2d(self.G,opts.out_channels,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1)
#         )
#         #init
#         for para in self.modules():
#             if isinstance(para,nn.Conv2d):
#                 nn.init.kaiming_normal_(para.weight)
#                 if para.bias is not None:
#                     para.bias.data.zero_()
#
#     def forward(self,x):
#         #f-1
#         f__1 = self.SFE1(x)
#         out  = self.SFE2(f__1)
#         RDB_outs = []
#         for i in range(self.D):
#             out = self.RDBS[i](out)
#             RDB_outs.append(out)
#         out = torch.cat(RDB_outs,1)
#         out = self.GFF(out)
#         out = f__1+out
#         return self.up_net(out)
#
# if __name__ == "__main__