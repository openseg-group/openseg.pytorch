##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Reproduce model writed by RainbowSecret
## Created by: Jianyuan Guo
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper

   
class Edge_Module(nn.Module):
    def __init__(self, mid_fea, out_fea, bn_type=None, factor=1):
        super(Edge_Module, self).__init__()
        
        self.conv1 =  nn.Sequential(
            nn.Conv2d(factor*256, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(mid_fea, bn_type=bn_type),
            ) 
        self.conv2 =  nn.Sequential(
            nn.Conv2d(factor*512, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(mid_fea, bn_type=bn_type),
            )  
        self.conv3 =  nn.Sequential(
            nn.Conv2d(factor*1024, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(mid_fea, bn_type=bn_type),
            )
        
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea*3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)
        
    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        
        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)        
        
        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)         
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge = self.conv5(edge)
         
        return edge, edge_fea