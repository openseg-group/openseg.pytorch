##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pdb
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.utils.helpers.offset_helper import DTOffsetConfig
from lib.models.backbones.hrnet.hrnet_backbone import BasicBlock


class SegFix_HRNet(nn.Module):
    def __init__(self, configer):
        super(SegFix_HRNet, self).__init__()
        self.configer = configer
        self.backbone = BackboneSelector(configer).get_backbone()
        backbone_name = self.configer.get('network', 'backbone')
        width = int(backbone_name[-2:])
        if 'hrnet2x' in backbone_name:
            in_channels = width * 31
        else:
            in_channels = width * 15

        num_masks = 2
        num_directions = DTOffsetConfig.num_classes

        mid_channels = 256

        self.dir_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            ModuleHelper.BNReLU(mid_channels,
                                bn_type=self.configer.get(
                                    'network', 'bn_type')),
            nn.Conv2d(mid_channels,
                      num_directions,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            ModuleHelper.BNReLU(mid_channels,
                                bn_type=self.configer.get(
                                    'network', 'bn_type')),
            nn.Conv2d(mid_channels,
                      num_masks,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        for i in range(1, len(x)):
            x[i] = F.interpolate(x[i],
                                 size=(h, w),
                                 mode='bilinear',
                                 align_corners=True)

        feats = torch.cat(x, 1)
        mask_map = self.mask_head(feats)
        dir_map = self.dir_head(feats)
        return mask_map, dir_map
