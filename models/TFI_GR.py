import torch
import torch.nn as nn
import torch.nn.functional as F
# from .resnet import resnet18
from torchvision.models.resnet import resnet18,resnet34


class Backbone_resnet(nn.Module):
    def __init__(self,backbone):
        super(Backbone_resnet, self).__init__()

        if backbone == 'resnet18':
            self.net = resnet18(pretrained=False)
            del self.net.avgpool
            del self.net.fc
        elif backbone == 'resnet34':
            self.net = resnet34(pretrained=False)
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
        return x, c1, c2, c3, c4



class TemporalFeatureInteractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureInteractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # difference enhance
        x_sub = self.conv_sub(torch.abs(x1 - x2))
        x1 = self.conv_diff_enh1(x1.mul(x_sub) + x1)
        x2 = self.conv_diff_enh2(x2.mul(x_sub) + x2)
        # fusion
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv_cat(x_f)
        x = x_sub + x_f
        x = self.conv_dr(x)
        return x

# if __name__ == "__main__":
#     x = torch.randn(1, 64, 16, 16)
#     y = torch.randn(1, 64, 16, 16)
#     net1 = TemporalFeatureInteractionModule(64)
#     x1 = net1(x,y)
#     #print(x1)
#     print(x1.shape)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChangeInformationExtractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(ChangeInformationExtractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # print(self.in_d)
        # print(self.out_d)
        self.ca = ChannelAttention(self.in_d * 4, ratio=16)
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d * 4, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.pools_sizes = [2, 4, 8]
        self.conv_pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[1], stride=self.pools_sizes[1]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[2], stride=self.pools_sizes[2]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, d5, d4, d3, d2):
        # upsampling
        d5 = F.interpolate(d5, d2.size()[2:], mode='bilinear', align_corners=True)
        d4 = F.interpolate(d4, d2.size()[2:], mode='bilinear', align_corners=True)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        # fusion
        x = torch.cat([d5, d4, d3, d2], dim=1)
        x_ca = self.ca(x)
        x = x * x_ca
        x = self.conv_dr(x)

        # feature = x[0:1, 0:64, 0:64, 0:64]
        # vis.visulize_features(feature)

        # pooling
        d2 = x
        d3 = self.conv_pool1(x)
        d4 = self.conv_pool2(x)
        d5 = self.conv_pool3(x)
        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)
        # print(d5.shape)
        return d5, d4, d3, d2


class GuidedRefinementModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(GuidedRefinementModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_d5 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d4 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p):
        # feature refinement
        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)
        # print(d5.shape)
        # print(d2_p.shape)
        # print(d3_p.shape)
        # print(d4_p.shape)
        # print(d5_p.shape)
        d5 = self.conv_d5(d5_p + d5)
        d4 = self.conv_d4(d4_p + d4)
        d3 = self.conv_d3(d3_p + d3)
        d2 = self.conv_d2(d2_p + d2)

        return d5, d4, d3, d2


class Decoder(nn.Module):
    def __init__(self, in_d, out_d):
        super(Decoder, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.conv_sum1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_sum2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_sum3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=False)

    def forward(self, d5, d4, d3, d2):

        d5 = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)
        d4 = self.conv_sum1(d4 + d5)
        d4 = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)
        d3 = self.conv_sum1(d3 + d4)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)
        d2 = self.conv_sum1(d2 + d3)

        mask = self.cls(d2)

        return mask


class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=2):
        super(BaseNet, self).__init__()
        self.backbone = Backbone_resnet(backbone='resnet18')
        self.mid_d = 64
        self.TFIM5 = TemporalFeatureInteractionModule(512, self.mid_d)
        self.TFIM4 = TemporalFeatureInteractionModule(256, self.mid_d)
        self.TFIM3 = TemporalFeatureInteractionModule(128, self.mid_d)
        self.TFIM2 = TemporalFeatureInteractionModule(64, self.mid_d)

        self.CIEM1 = ChangeInformationExtractionModule(self.mid_d, output_nc)
        self.GRM1 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.CIEM2 = ChangeInformationExtractionModule(self.mid_d, output_nc)
        self.GRM2 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.decoder = Decoder(self.mid_d, output_nc)

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        # feature difference
        d5 = self.TFIM5(x1_5, x2_5)  # 1/32
        d4 = self.TFIM4(x1_4, x2_4)  # 1/16
        d3 = self.TFIM3(x1_3, x2_3)  # 1/8
        d2 = self.TFIM2(x1_2, x2_2)  # 1/4

        # change information guided refinement 1
        d5_p, d4_p, d3_p, d2_p = self.CIEM1(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM1(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)
        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)
        # print(d5.shape)
        # change information guided refinement 2
        d5_p, d4_p, d3_p, d2_p = self.CIEM2(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM2(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # decoder
        mask = self.decoder(d5, d4, d3, d2)

        mask = F.interpolate(mask, x1.size()[2:], mode='bilinear', align_corners=True)

        mask = torch.sigmoid(mask)

        return mask


if __name__ == '__main__':
    x1 = torch.rand(4,3,256,256)
    x2 = torch.rand(4,3,256,256)
    model = BaseNet()

    from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(x1, x2)
    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))
