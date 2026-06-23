import math

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from matplotlib import pyplot as plt
from timm.models.layers import trunc_normal_
from torch.nn import Parameter

# from models.dynamic_conv import DynamicConv, DynamicConv1



class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


# Channel-wise Correlation
# class CCorrM(nn.Module):
#     def __init__(self, all_channel):
#         super(CCorrM, self).__init__()
#         self.linear_e = nn.Linear(all_channel, all_channel, bias=False) #weight
#         self.channel = all_channel
#         self.conv1 = DSConv3x3(all_channel, all_channel, stride=1)
#         self.conv2 = DSConv3x3(all_channel, all_channel, stride=1)
#
#     def forward(self,x):  # exemplar: f1, query: f2
#         query = x[0]
#         exemplar = x[1]
#         fea_size = query.size()[2:]
#         exemplar = F.interpolate(exemplar, size=fea_size, mode="bilinear", align_corners=True)
#         all_dim = fea_size[0] * fea_size[1]
#         exemplar_flat = exemplar.view(-1, self.channel, all_dim)  # N,C1,H,W -> N,C1,H*W
#         query_flat = query.view(-1, self.channel, all_dim)  # N,C2,H,W -> N,C2,H*W
#         exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batchsize x dim x num, N,H*W,C1
#         exemplar_corr = self.linear_e(exemplar_t)  # batchsize x dim x num, N,H*W,C1
#         A = torch.bmm(query_flat, exemplar_corr)  # ChannelCorrelation: N,C2,H*W x N,H*W,C1 = N,C2,C1
#
#         A1 = F.softmax(A.clone(), dim=2)  # N,C2,C1. dim=2 is row-wise norm. Sr
#         # B = F.softmax(torch.transpose(A, 1, 2), dim=2)  # N,C1,C2 column-wise norm. Sc
#         query_att = torch.bmm(A1, exemplar_flat).contiguous()  # N,C2,C1 X N,C1,H*W = N,C2,H*W
#         # exemplar_att = torch.bmm(B, query_flat).contiguous()  # N,C1,C2 X N,C2,H*W = N,C1,H*W
#
#         # exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C1,H*W -> N,C1,H,W
#         # exemplar_out = self.conv1(exemplar_att + exemplar)
#
#         query_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])  # N,C2,H*W -> N,C2,H,W
#         query_out = self.conv1(query_att + query)
#         out = query_out + exemplar
#         return out


# Edge-based Enhancement Unit (EEU)
class EEU(nn.Module):
    def __init__(self, in_channel):
        super(EEU, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)

    def forward(self, x):
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # edge = self.PReLU(edge)
        out = weight * x + x
        return out


# Edge Self-Alignment Module (ESAM)
class ESAM(nn.Module):
    def __init__(self, in_channel):
        super(ESAM, self).__init__()
        self.eeu = EEU(in_channel)
    def forward(self, t):  # x1 16*144*14; x2 24*72*72
        t_2 = self.eeu(t)
        return t_2  # (24*2)*144*144


# class Fuse(nn.Module):
#     def __init__(self, in_channel):
#         super(Fuse, self).__init__()
#         self.esam = ESAM(in_channel)
#         self.DSMM = DSMM(in_channel)
#         self.mam  = MAM2(in_channel)
#     def forward(self,x):
#         rgb = x[0]
#         t = x[1]
#         t1 = self.esam(t)
#         rgb1 = self.DSMM(rgb)
#         x = [rgb1,t1]
#         final = self.mam(x)
#         return final


class Fuse1(nn.Module):
    def __init__(self, in_channel):
        super(Fuse1, self).__init__()
        # self.esam = ESAM(in_channel)
        # self.DSMM = DSMM1(in_channel)
        self.mam = MAM2(in_channel)
        self.fuse = FRM(in_channel)
        # self.dy = DynamicConv1(in_channel, in_channel, 3, 1, 1)
        # self.fus1 = Conv(in_channel * 2, in_channel, 1, 1, 0)
    def forward(self,x):
        rgb = x[0]
        t = x[1]
        # t1 = self.esam(t)
        # rgb1 = self.DSMM(rgb,t)
        x = [rgb, t]
        # x = [t, rgb]
        # x = [t, rgb]
        gr = self.mam(x)
        # final = gr+t
        # map_rgb = torch.unsqueeze(torch.mean(final, 1), 1)
        # score2 = F.interpolate(map_rgb, size=(128, 128), mode="bilinear", align_corners=True)
        # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
        # depth = (score2 - score2.min()) / (score2.max() - score2.min())
        # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        # plt.imshow(feature_img)
        # plt.show()
        # plt.savefig("2.png")
        # gt = self.mam(x)
        # dy_rgb = self.dy(gr,t)
        final = self.fuse(gr, t)
        # final = gr + t
        # final = self.fuse(rgb, gt)
        # fuse = self.fus1(torch.cat([gr, t], dim=1))
        # final = gr+t
        # map_rgb = torch.unsqueeze(torch.mean(final, 1), 1)
        # score2 = F.interpolate(map_rgb, size=(80, 80), mode="bilinear", align_corners=True)
        # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
        # depth = (score2 - score2.min()) / (score2.max() - score2.min())
        # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
        # plt.imshow(feature_img)
        # plt.show()
        # plt.savefig("1.png")
        return final


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        # x = torch.cat((x1, x2), dim=1)
        # avg1 = self.avg_pool(x1).view(B, self.dim)
        avg1 = torch.mean(x1, dim=[2, 3], keepdim=True).view(B, self.dim)
        avg2 = torch.mean(x2, dim=[2, 3], keepdim=True).view(B, self.dim)
        # avg2 = self.avg_pool(x2).view(B, self.dim)
        max1 = self.max_pool(x1).view(B, self.dim)
        max2 = self.max_pool(x2).view(B, self.dim)
        avg = avg1+avg2
        max = max1+max2
        y = torch.cat((max, avg), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class FRM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FRM, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def forward(self, x1, x2):
    #     channel_weights = self.channel_weights(x1, x2)
    #     # out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
    #     x1 = x1 + self.lambda_c * channel_weights[0] * x1
    #     # out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
    #     x2 = x2 + self.lambda_c * channel_weights[1] * x2
    #     spatial_weights = self.spatial_weights(x1, x2)
    #     out_x1 = x1 + self.lambda_s * spatial_weights[0] * x1
    #     out_x2 = x2 + self.lambda_s * spatial_weights[1] * x2
    #     out = out_x1 + out_x2
    #     return out
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1+ self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0]* x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1+out_x2  

if __name__ == "__main__":
    torch.cuda.set_device(1)
    x = torch.rand(2,256,80,80).cuda()
    y = torch.rand(2,256,80,80).cuda()
  
    frm = FRM(256).cuda()
    out = frm(x,y)
    print(out.shape)