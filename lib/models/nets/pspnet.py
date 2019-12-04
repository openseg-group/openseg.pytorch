#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Pytorch implementation of PSP net Synchronized Batch Normalization
# this is pytorch implementation of PSP resnet101 (syn-bn) version


import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class PSPNet(nn.Sequential):
    """
    Pyramid Scene Parsing Network, CVPR2017
    """
    def __init__(self, configer):
        super(PSPNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        num_features = self.backbone.get_num_features()
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(num_features // 4, self.num_classes, 1, 1, 0)
        )
        from lib.models.modules.psp_block import PSPModule
        self.ppm = PSPModule(2048, 512, bn_type=self.configer.get('network', 'bn_type'))
        self.cls = nn.Conv2d(512, self.num_classes, kernel_size=1)

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn(x[-2])
        x = self.ppm(x[-1])
        x = self.cls(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


if __name__ == '__main__':
    i = torch.Tensor(1,3,512,512).cuda()
    model = PSPNet(num_classes=19).cuda()
    model.eval()
    o, _ = model(i)
    print(o.size())
    print(_.size())
