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

