##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class DeepLabV3(nn.Module):
    """
    Rethinking Atrous Convolution for Semantic Image Segmentation
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(DeepLabV3, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        from lib.models.modules.aspp_block import ASPPModule
        self.head = nn.Sequential(ASPPModule(in_channels[1], 256, 256, bn_type=self.configer.get('network', 'bn_type')),
                                  nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn(x[-2])
        x = self.head(x[-1])
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


class DeepLabV3_CRF(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(DeepLabV3_CRF, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        from lib.models.modules.aspp_block import ASPPModule
        self.aspp_head = ASPPModule(in_channels[1], 256, 256, bn_type=self.configer.get('network', 'bn_type'))
        self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        from lib.models.modules.conv_crf_block import train_config, GaussCRF
        self.crf_head = GaussCRF(train_config, self.configer.get('train', 'data_transform', 'input_size')//8, 19)

        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn(x[-2])
        x_aspp = self.aspp_head(x[-1])
        x = self.cls_head(x_aspp)
        x_crf = self.crf_head(x_aspp, x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x_crf = F.interpolate(x_crf, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x, x_crf