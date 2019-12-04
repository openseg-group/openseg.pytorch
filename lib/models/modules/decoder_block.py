##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Jianyuan Guo, Rainbowsecret
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper


class Decoder_Module_Plus(nn.Module):

    def __init__(self, 
                 in_plane=256, 
                 mid_plane=256, 
                 out_plane=256,
                 bn_type=None):
        super(Decoder_Module_Plus, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_plane, mid_plane, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(mid_plane, bn_type=bn_type),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_plane+256, out_plane, kernel_size=5, padding=2, bias=False),
            ModuleHelper.BNReLU(out_plane, bn_type=bn_type),
            nn.Conv2d(out_plane, out_plane, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_plane, bn_type=bn_type),
            )
        

    def forward(self, x_low, x_high):
        _, _, h, w = x_low.size()
        x_low = self.conv1(x_low)
        x_high = F.interpolate(x_high, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([x_low, x_high], dim=1)
        x = self.conv2(x)
        return x  


class Decoder_Module(nn.Module):

    def __init__(self, bn_type=None, inplane1=512, inplane2=256, outplane=128):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplane2, 48, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(48, bn_type=bn_type),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, outplane, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(outplane, bn_type=bn_type),
            nn.Conv2d(outplane, outplane, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(outplane, bn_type=bn_type),
            )
        

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        return x  


class CE2P_Decoder_Module(nn.Module):

    def __init__(self, num_classes, dropout=0, bn_type=None, inplane1=512, inplane2=256):
        super(CE2P_Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplane2, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(48, bn_type=bn_type),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Dropout2d(dropout),
            )
        
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x  


class Separable_Decoder_Module(nn.Module):

    def __init__(self, num_classes, bn_type=None, inplane1=512, inplane2=128):
        super(Separable_Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplane2, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(48, bn_type=bn_type),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 304, kernel_size=3, stride=1, padding=1, dilation=1, groups=304, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(304),
            nn.Conv2d(304, 256, 1, 1, 0, 1, 1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, groups=256, bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(256),
            nn.Conv2d(256, 256, 1, 1, 0, 1, 1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            )
        

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        return x  


class Decoder_Module_SA(nn.Module):

    def __init__(self, num_classes, dropout=0, bn_type=None, inplane1=512, inplane2=128):
        super(Decoder_Module_SA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplane1, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplane2, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(48, bn_type=bn_type),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            )

        from lib.models.modules.conv_sa_block import LocalSA
        self.local_sa = LocalSA(in_channels=256, 
                                key_channels=128, 
                                value_channels=128, 
                                out_channels=256, 
                                kernel_size=3, 
                                bn_type=bn_type)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )       

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        x_sa = self.local_sa(x)
        seg = self.conv4((torch.cat([x, x_sa], 1)))
        return seg


class Decoder_Module_2x(nn.Module):

    def __init__(self, num_classes, dropout=0, hr_channels=128, bn_type=None):
        super(Decoder_Module_2x, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hr_channels, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(48, bn_type=bn_type),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Dropout2d(dropout),
            )
        
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg  


class Decoder_Module_2x_v2(nn.Module):

    def __init__(self, num_classes, dropout=0, hr_channels=128, bn_type=None):
        super(Decoder_Module_2x_v2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hr_channels, 48, kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(48, bn_type=bn_type),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=bn_type),
            nn.Dropout2d(dropout),
            )
        
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg  


class Decoder_Module_2x_v3(nn.Module):

    def __init__(self, num_classes, dropout=0, hr_channels=128, bn_type=None):
        super(Decoder_Module_2x_v3, self).__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(hr_channels, 128, kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(128, bn_type=bn_type),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=bn_type),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=bn_type),
            nn.Dropout2d(dropout),
            )
        
        self.conv4 = nn.Conv2d(512, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg  


class Decoder_Module_2x_v4(nn.Module):

    def __init__(self, num_classes, dropout=0, hr_channels=128, bn_type=None):
        super(Decoder_Module_2x_v4, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=bn_type),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=bn_type),
            nn.Dropout2d(dropout),
            )
        
        self.conv4 = nn.Conv2d(512, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg  