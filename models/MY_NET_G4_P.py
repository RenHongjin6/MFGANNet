"""Backbone+GSA+FFM+FEM+CFM"""
"""
GSA:全局语义聚合模块
FFM：特征融合模块
FEM：特征强化模块
CFM：跨尺度融合模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18,resnet34
from torch.nn import init
from torchvision import models

class Backbone_resnet(nn.Module):
    def __init__(self,backbone):
        super(Backbone_resnet, self).__init__()

        if backbone == 'resnet18':
            self.net = resnet18(pretrained=True)
            # self.net = resnet18(weights = models.ResNet18_Weights.DEFAULT)

            del self.net.avgpool
            del self.net.fc
        elif backbone == 'resnet34':
            self.net = resnet34(pretrained=True)
            # self.net = resnet34(models.ResNet34_Weights.DEFAULT)
            del self.net.avgpool
            del self.net.fc
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def forward(self,x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        c1 = self.net.layer1(x)
        c2 = self.net.layer2(c1)
        c3 = self.net.layer3(c2)
        c4 = self.net.layer4(c3)
        return c1, c2, c3, c4


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class MultiConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiConv, self).__init__()

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse_attn(x)



class PPM(nn.Module):
    def __init__(self,in_dim=512,bins=[1,2,3,6]):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim,in_dim,kernel_size=1,bias=False),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self,x):
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(F.interpolate(f(x),x_size[2:],mode='bilinear',align_corners=True))
        return out[0]+out[1]+out[2]+out[3]


class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Double_conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FFM(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(FFM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # self.conv1x1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=1),
        #     nn.Sigmoid(),
        # )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

        )

        self.init_weight()

    def forward(self,x1,x2):
        """x1 _ cat , x2 _ sub"""
        x1 = self.avgpool(x1)
        x1_ = self.conv1(x1)
        # x1_ = self.conv1x1(x1)
        x2_ = self.conv3x3(x2)
        out = torch.mul(x1_,x2_)
        return out + x2

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Spatial_Attention(nn.Module):
    """空间注意力模块"""
    def __init__(self,kernel_size = 7):
        super(Spatial_Attention, self).__init__()
        assert kernel_size in (3,7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2,1,kernel_size ,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        out = torch.cat([avg_out,max_out],dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Channel_Attention(nn.Module):
    def __init__(self,in_ch,ration = 16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_ch,in_ch//ration,kernel_size=1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_ch//ration,in_ch,kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class attention(nn.Module):
    def __init__(self, in_ch):
        super(attention, self).__init__()
        self.sa = Spatial_Attention()
        self.ca = Channel_Attention(in_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2,in_ch,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.ca(x)
        x1_ = torch.mul(x,x1)
        x2 = self.sa(x1_)
        x2_ = torch.mul(x,x2)
        x2_ = self.sigmoid(x2_)
        x3 = torch.mul(x2_,x2)
        out = torch.cat([x3,x],dim=1)
        out = self.conv(out) + x

        return out


class MY_NET(nn.Module):
    def __init__(self,num_classes=2):
        super(MY_NET, self).__init__()

        self.net = Backbone_resnet(backbone='resnet34')

        """上采样"""
        self.up4 = nn.Sequential(
            Double_conv(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up3 = nn.Sequential(
            Double_conv(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up2 = nn.Sequential(
            Double_conv(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )

        self.up1 = nn.Sequential(
            Double_conv(64, 64),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), )

        """下采样"""
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)

        self.down1_2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=1,stride=1,padding=0,bias=False),
            nn.AvgPool2d(2,stride=2,padding=0)
        )
        self.down2_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2, padding=0)
        )
        self.down3_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2, padding=0)
        )

        self.layer1_1x1 = Double_conv1x1(64, 64)
        self.layer2_1x1 = Double_conv1x1(128, 128)
        self.layer3_1x1 = Double_conv1x1(256, 256)
        self.layer4_1x1 = Double_conv1x1(512, 512)


        self.sp1 = attention(64)
        self.sp2 = attention(128)
        self.sp3 = attention(256)
        self.sp4 = attention(512)

        self.fuse = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.fuse_ = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.ppm = PPM()

        self.output =nn.Sequential(

            nn.Conv2d(64,num_classes,kernel_size=3,stride=1,padding=1,bias=False),
        )

    def forward(self,x1,x2):
        h, w = x1.shape[2:]
        x1_layer1,x1_layer2,x1_layer3,x1_layer4 = self.net(x1)
        x2_layer1,x2_layer2,x2_layer3,x2_layer4 = self.net(x2)

        layer4_out = x1_layer4 + x2_layer4
        layer4_out = self.ppm(layer4_out)

        """cat_branch"""
        cat_layer1 = torch.cat([x1_layer1,x2_layer1],dim=1)

        cat_layer2 = torch.cat([x1_layer2,x2_layer2],dim=1)

        cat_layer3 = torch.cat([x1_layer3,x2_layer3],dim=1)

        cat_layer4 = torch.cat([x1_layer4,x2_layer4],dim=1)

        """sub_branch"""
        sub_layer1 = torch.abs(x1_layer1 - x2_layer1)

        sub_layer2 = torch.abs(x1_layer2 - x2_layer2)

        sub_layer3 = torch.abs(x1_layer3 - x2_layer3)

        sub_layer4 = torch.abs(x1_layer4 - x2_layer4)

        """跨尺度融合"""
        down1 = self.down1(sub_layer1)

        down2 = F.interpolate(self.down2(sub_layer2),size=sub_layer1.size()[2:], mode='bilinear', align_corners=True)

        down3 = F.interpolate(self.down3(sub_layer3), size=sub_layer1.size()[2:], mode='bilinear',align_corners=True)

        down4 = F.interpolate(self.down4(sub_layer4), size=sub_layer1.size()[2:], mode='bilinear',align_corners=True)

        down1234 = self.fuse(torch.cat([down1+down2, down2+down1+down3, down3+down2+down4, down4+down3], dim=1))
        down1234_ = self.fuse_(torch.cat([down1+down2, down2+down1+down3, down3+down2+down4, down4+down3], dim=1))

        cat_layer1 = torch.cat([cat_layer1,down1234],1)
        c_layer1 = self.ffm1(cat_layer1,sub_layer1)
        layer1 = self.sp1(c_layer1)
        layer1 = self.layer1_1x1(layer1)

        cat_layer2 = torch.cat([cat_layer2,down1234_],1)
        c_layer2 = self.ffm2(cat_layer2,sub_layer2) + self.down1_2(layer1)
        layer2 = self.sp2(c_layer2)
        layer2 = self.layer2_1x1(layer2)

        c_layer3 = self.ffm3(cat_layer3,sub_layer3) + self.down2_3(layer2)
        layer3 = self.sp3(c_layer3)
        layer3 = self.layer3_1x1(layer3)

        c_layer4 = self.ffm4(cat_layer4,sub_layer4) + self.down3_4(layer3)
        layer4 = self.sp4(c_layer4)
        layer4 = self.layer4_1x1(layer4)

        out4 = layer4+layer4_out
        up4 = self.up4(out4)

        out3 = layer3+up4
        up3 = self.up3(out3)

        out2 = layer2+up3
        up2 = self.up2(out2)

        out1 = layer1+up2
        up1 = self.up1(out1)

        output = self.output(up1)
        return output


if __name__ == '__main__':
    x1 = torch.rand(4,3,256,256)
    x2 = torch.rand(4,3,256,256)
    model = MY_NET()
    a = model(x1,x2)
    print(a.shape)


    from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(x1, x2)
    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))

    # import time
    # if torch.cuda.is_available():
    #     model = model.cuda()  # .half()  #HALF seems to be doing slower for some reason
    #     x1 = x1.cuda()  # .half()
    #     x2 = x2.cuda()
    #
    # time_train = []
    # i = 0
    # # model.load_state_dict(torch.load("../Testmodel_List/KR94187_Portrait_98/result/Dnc_C3Portrait/model_266.pth",
    # #                       map_location=torch.device(device='cpu')))
    # # 0.273
    # while (i < 20):
    #     # for step, (images, labels, filename, filenameGt) in enumerate(loader):
    #
    #     start_time = time.time()
    #
    #     inputs1 = torch.autograd.Variable(x1)
    #     inputs2 = torch.autograd.Variable(x2)
    #     with torch.no_grad():
    #         outputs = model(x1,x2)
    #
    #     # preds = outputs.cpu()
    #     # if torch.cuda.is_available():
    #     #     torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
    #
    #     if i != 0:  # first run always takes some time for setup
    #         fwt = time.time() - start_time
    #         time_train.append(fwt)
    #         print("Forward time per img (b=%d): %.3f (Mean: %.3f)" % (
    #            1, fwt / 1, sum(time_train) / len(time_train) /1))
    #
    #     time.sleep(1)  # to avoid overheating the GPU too much
    #     i += 1
