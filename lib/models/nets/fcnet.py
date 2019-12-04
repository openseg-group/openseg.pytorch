##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class FcnNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FcnNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        elif "mobilenetv2" in self.configer.get('network', 'backbone'):
            in_channels = [160, 320]
        else:
            in_channels = [1024, 2048]
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        if "mobilenetv2" in self.configer.get('network', 'backbone'):
            self.cls_head = nn.Sequential(
                nn.Conv2d(in_channels[1], 256, kernel_size=3, stride=1, padding=1),
                ModuleHelper.BNReLU(256, bn_type=self.configer.get('network', 'bn_type')),
                nn.Dropout2d(0.10),
                nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.dsn_head = nn.Sequential(
                nn.Conv2d(in_channels[0], 128, kernel_size=3, stride=1, padding=1),
                ModuleHelper.BNReLU(128, bn_type=self.configer.get('network', 'bn_type')),
                nn.Dropout2d(0.10),
                nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn_head(x[-2])
        x = self.cls_head(x[-1])
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return aux_x, x


class FcnNet_wo_dsn(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FcnNet_wo_dsn, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        elif "mobilenetv2" in self.configer.get('network', 'backbone'):
            in_channels = [160, 320]
        else:
            in_channels = [1024, 2048]
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        if "mobilenetv2" in self.configer.get('network', 'backbone'):
            self.cls_head = nn.Sequential(
                nn.Conv2d(in_channels[1], 256, kernel_size=3, stride=1, padding=1),
                ModuleHelper.BNReLU(256, bn_type=self.configer.get('network', 'bn_type')),
                nn.Dropout2d(0.10),
                nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x = self.cls_head(x[-1])
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x