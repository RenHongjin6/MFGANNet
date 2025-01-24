import torch
from torch import nn
import warnings
#from eca_module import eca_layer
import torch.nn.functional as F
import math

warnings.filterwarnings(action='ignore')


####################################################################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__() # super类的作用是继承的时候，调用含super的哥哥的基类__init__函数。
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False) # 一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size() # b代表b个样本，c为通道数，h为高度，w为宽度

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)
        # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算
        return x * y.expand_as(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,k_size=3):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.eca(out)
        # print(out.shape)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, k_size=3):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def _resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def _resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def _resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def _resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


####################################################################################################

class resnet18(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = _resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class resnet34(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = _resnet34(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class resnet50(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = _resnet50()
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class resnet101(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = _resnet101(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


def build_contextpath(name='resnet50'):
    model = {
        'resnet34': resnet34(pretrained=False),
        'resnet18': resnet18(pretrained=False),
        'resnet50': resnet50(pretrained=False),
        'resnet101': resnet101(pretrained=False)
    }
    return model[name]


####################################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Spatial_path(torch.nn.Module):
    """Assimilation Network"""
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=6, out_channels=128)
        self.convblock2 = ConvBlock(in_channels=128, out_channels=256)
        self.convblock3 = ConvBlock(in_channels=256, out_channels=512)
        self.init_weight()

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Differentiation_path(torch.nn.Module):
    """Differece Network"""
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)
        self.init_weight()

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class AttentionRefinementModule(torch.nn.Module):
    """Fusion Network"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.init_weight()

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BGALayer4(nn.Module):#4倍
    """Feature Aggregation Network"""
    def __init__(self):
        super(BGALayer4, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                )
        self.left2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
                                )
        self.right1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(512),
                                )
        self.right2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                              )

        ##TODO: does this really has no relu?


        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), # not shown in paper
                                )
        self.init_weight()


    def forward(self, dx, cx2):#x_d.Size([2, 128, 64, 64]).x_s.Size([2, 128, 16, 16])  x_different.Size([2, 128, 64, 64])

        dsize = dx.size()[2:]
        left1 = self.left1(dx)#torch.Size([2, 128, 64, 64])
        left2 = self.left2(dx)#torch.Size([2, 128, 16, 16])
        right1 = self.right1(cx2)#torch.Size([2, 128, 16, 16])
        right2 = self.right2(cx2)#torch.Size([2, 128, 16, 16])
        right1 = F.interpolate(right1, size=dsize, mode='bilinear', align_corners=True)
        #右1上采用4倍才能和左1相加torch.Size([2, 128, 64, 64])
        left = left1 * torch.sigmoid(right1)#torch.Size([2, 128, 64, 64])
        right = left2 * torch.sigmoid(right2)#torch.Size([2, 128, 16, 16])
        right = F.interpolate(right, size=dsize, mode='bilinear', align_corners=True)
        #左2和右2合并上采用4倍然后输出torch.Size([2, 128, 64, 64])
        out = self.conv(left + right )#torch.Size([2, 128, 64, 64])
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BGALayer2(nn.Module):#2倍

    def __init__(self):
        super(BGALayer2, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, stride=1,padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1, groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, stride=1,padding=0, bias=False),
        )

        ##TODO: does this really has no relu?


        self.conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.init_weight()


    def forward(self, sx, cx1):#x_d.Size([2, 128, 64, 64]).x_s.Size([2, 128, 16, 16])  x_different.Size([2, 128, 64, 64])

        dsize = sx.size()[2:]
        left1 = self.left1(sx)#torch.Size([2, 128, 64, 64])
        left2 = self.left2(sx)#torch.Size([2, 128, 16, 16])
        right1 = self.right1(cx1)#torch.Size([2, 128, 16, 16])
        right2 = self.right2(cx1)#torch.Size([2, 128, 16, 16])
        right1 = F.interpolate(
            right1, size=dsize, mode='bilinear', align_corners=True)#右1上采用4倍才能和左1相加torch.Size([2, 128, 64, 64])
        left = left1 * torch.sigmoid(right1)#torch.Size([2, 128, 64, 64])
        right = left2 * torch.sigmoid(right2)#torch.Size([2, 128, 16, 16])
        right = F.interpolate(
            right, size=dsize, mode='bilinear', align_corners=True)#左2和右2合并上采用4倍然后输出torch.Size([2, 128, 64, 64])


        out = self.conv(left + right )#torch.Size([2, 128, 64, 64])


        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # self.in_channels = 3584
        # self.in_channels = 3328
        self.in_channels = 768
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.init_weight()

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)#torch.Size([1, 1280, 64, 64])
        feature = self.convblock(x)#torch.Size([1, 2, 64, 64])
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)#torch.Size([1, 2, 64, 64])
        x = torch.add(x, feature)#torch.Size([1, 2, 64, 64])
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class MFGANnet(torch.nn.Module):
    def __init__(self, num_classes=2, context_path='resnet34'):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()#(1,256,64,64)
        self.differentiation_path = Differentiation_path()#(1,256,64,64)

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module
        self.attention_refinement_module1 = AttentionRefinementModule(256, 256) # resnet 18 / 34
        self.attention_refinement_module2 = AttentionRefinementModule(512, 512) # resnet 18 / 34
        # self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)  # resnet 50 / 101
        # self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)  # resnet 50 / 101
        self.bga2 = BGALayer2()
        self.bga4 = BGALayer4()



        # build feature fusion module
        self.feature_fusion_module = FeatureFusionModule(num_classes)

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        # supervision block
        self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1) #resnet 18 / 34
        self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1) # resnet 18 / 34
        # self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)  # resnet 50 / 101
        # self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)  # resnet 50 / 101

    def forward(self, x1, x2):
        input = torch.cat((x1, x2), 1)
        # output of spatial path
        sx = self.saptial_path(input) # (1, 256, input/8, input/8)([1, 256, 64, 64])
        # print(sx.shape)
        dx1 = self.differentiation_path(x1)#([1, 512, 64, 64])
        # print(dx1.shape)
        dx2 = self.differentiation_path(x2)#([1, 512, 64, 64])
        dx = dx1 - dx2##([1, 512, 64, 64])
        # print(dx1.shape)


        # output of context path
        cx1, cx2, tail = self.context_path(input) # ([1, 256, 32, 32])  tail([1, 512, 1, 1])
        # print(cx1.shape)
        # print(cx2.shape)
        # print(tail.shape)
        cx1 = self.attention_refinement_module1(cx1)#[1, 256, 32, 32])
        cx2 = self.attention_refinement_module2(cx2) # [1, 512, 16, 16])
        cx2 = torch.mul(cx2, tail)#[1, 512, 16, 16])
        # print(cx1.shape)
        # print(cx2.shape)
        # print(tail.shape)

        feat4 = self.bga4(sx, cx2)#torch.Size([1, 512, 64, 64])
        # print('a',feat4.shape)
        feat2 = self.bga2(dx, cx1)#torch.Size([1, 256, 64, 64])
        # print('b',feat2.shape)


        # upsampling
        cx3 = torch.nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear') # (1, 1024, input/8, input/8)([1, 256, 64, 64])
        cx4 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear') # (1, 2048, input/8, input/8)([1, 512, 64, 64])

        # cx = torch.cat((feat2, feat4), dim=1) # (1, 3072, input/8, input/8)([1, 768, 64, 64])

        if self.training == True:
            cx3_sup = self.supervision1(cx3)
            cx4_sup = self.supervision2(cx4)
            cx3_sup = torch.nn.functional.interpolate(cx3_sup, scale_factor=8, mode='bilinear')
            cx4_sup = torch.nn.functional.interpolate(cx4_sup, scale_factor=8, mode='bilinear')

        # output of feature fusion module
        result = self.feature_fusion_module(feat2, feat4)#([1, 2, 64, 64])

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        if self.training == True:
            return result, cx3_sup, cx4_sup

        return result


if __name__ == '__main__':
    from models.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model = MFGANnet()

    x1 = torch.rand((2, 3, 512, 512))
    x2 = torch.rand((2, 3, 512, 512))

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()

    out = model_eval(x1, x2)

    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))