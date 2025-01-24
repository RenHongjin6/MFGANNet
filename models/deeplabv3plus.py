from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


###################################################################
class ConvBnReLU(nn.Sequential):
    """
    封装conv+bn+relu
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=True):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            'Conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        )
        self.add_module('BN', nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module('ReLU', nn.ReLU())

"""
全局平均池化+1*1卷积+双线性插值上采样
"""
class ImagePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        x_size = x.shape
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module('c0', ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1))
        for idx, rate in enumerate(rates):
            self.stages.add_module('c{}'.format(idx+1), ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate))
        self.stages.add_module('imagepool', ImagePool(in_channels, out_channels))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)

# if __name__ == '__main__':
#     aspp = ASPP(128, 256, rates=(6,12,18,24)).cuda()
#     input = torch.rand([2,128,28,28]).cuda()
#     output = aspp(input)
#     print(output.shape)
#######################################################################################################################
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, downsample):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4
        self.reduce = ConvBnReLU(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0, dilation=1, relu=True)
        self.conv3X3 = ConvBnReLU(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, relu=True)
        self.increase = ConvBnReLU(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, relu=False)
        self.shortcut =ConvBnReLU(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, relu=False)

    def forward(self, x):
        x_ = self.reduce(x)
        x_ = self.conv3X3(x_)
        x_ = self.increase(x_)
        if self.downsample:
            x_ += self.shortcut(x)
        else:
            x_ += x
        return F.relu(x_)

class ResLayer(nn.Sequential):
    def __init__(self, num_layers, in_channels, out_channels, stride, dialtion, multi_grids=None):
        super(ResLayer, self).__init__()
        if multi_grids is None:
            multi_grids = [1 for _ in range(num_layers)]
        else:
            assert num_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(num_layers):
            self.add_module(
                'block{}'.format(i+1), Bottleneck(
                    in_channels=(in_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    stride=(stride if i == 0 else 1),
                    dilation=dialtion * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


"""
The first conv layer
Note that the max pooling is different from both MSRA and FAIR ResNet.
"""


class Stem(nn.Sequential):
    def __init__(self, out_channels):
        super(Stem, self).__init__()
        self.add_module('conv1', ConvBnReLU(6, out_channels, kernel_size=7, stride=2, padding=3, dilation=1))
        self.add_module('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

########################################################################################################################


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_class=2):
        super(DeepLabV3Plus, self).__init__()

        n_blocks=[3,4,6,3]    # backbone=resnet101,num_eachlayer_bottleneck=[3,4,23,3]
        multi_grids=[1,2,4]    # 3x3Conv in layer5 each dilation is dilation*[1,2,4]

        self.layer1=Stem(64)   # 下采样X4
        self.layer2=ResLayer(n_blocks[0],64,256,stride=1,dialtion=1)
        self.layer3=ResLayer(n_blocks[1],256,512,stride=2,dialtion=1)    # 下采样X8
        self.layer4=ResLayer(n_blocks[2],512,1024,stride=1,dialtion=2)   # 在bottleneck中的3x3卷积一致采用空洞策略且p=d保证图像尺寸
        self.layer5=ResLayer(n_blocks[3],1024,2048,stride=1,dialtion=4,multi_grids=multi_grids)  # 空洞卷积大小设置为[4,8,16]

        atrous_rates=[6,12,18]  # 3x3Conv in ASPP each dilation is 6,12,18

        self.aspp=ASPP(2048,256,atrous_rates)
        self.fc1=ConvBnReLU((len(atrous_rates)+2)*256,256,kernel_size=1,stride=1,padding=0)

        # Decoder
        self.reduce=ConvBnReLU(256,48,kernel_size=1,stride=1,padding=0)
        # layer2的输出在进行下采样之前进行copy操作，与下采样之后的特征图以48/256的比率concat,最终输出304个通道

        self.fc2=nn.Sequential(
            ConvBnReLU(304,256,kernel_size=3,stride=1,padding=1),  # 图片尺寸不变
            ConvBnReLU(256,256,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(256,n_class,kernel_size=1,stride=1,padding=0)
        )

    def forward(self,x, y):
        x_ = torch.cat([x,y],1)
        h=self.layer1(x_)
        h=self.layer2(h)
        h_=self.reduce(h)
        h=self.layer3(h)
        h=self.layer4(h)
        h=self.layer5(h)
        h=self.aspp(h)
        h=self.fc1(h)
        h=F.interpolate(h, size=h_.shape[2:], mode="bilinear", align_corners=False)  # 将h恢复成h_的尺寸这里应该是X2
        h=torch.cat((h_,h),dim=1)
        h=self.fc2(h)
        h=F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)   # 将h恢复成x的尺寸这里应该是X4
        return h


if __name__=='__main__':
    image=torch.randn(1,3,256,256)
    net=DeepLabV3Plus(2)
    net.eval()
    out=net(image, image)
    print(out.shape)
    # 保存网络


