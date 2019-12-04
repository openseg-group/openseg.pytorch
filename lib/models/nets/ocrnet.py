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
from torch.nn import functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class SpatialOCRNet(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(SpatialOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )

        from lib.models.modules.spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module
        self.spatial_context_head = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512, 
                                                  key_channels=256, 
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0.05, 
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.conv_3x3(x[-1])
        context = self.spatial_context_head(x, x_dsn)
        x = self.spatial_ocr_head(x, context)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return  x_dsn, x


class ASPOCRNet(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(ASPOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]

        # we should increase the dilation rates as the output stride is larger
        from lib.models.modules.spatial_ocr_block import SpatialOCR_ASP_Module
        self.asp_ocr_head = SpatialOCR_ASP_Module(features=2048, 
                                                  hidden_features=256, 
                                                  out_features=256,
                                                  num_classes=self.num_classes,
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.asp_ocr_head(x[-1], x_dsn)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return  x_dsn, x
