#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: speedinghzl
# deeplabv3 res101 (synchronized BN version)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), bn_type=None):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, bn_type) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, bn_type):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = ModuleHelper.BNReLU(out_features, bn_type=bn_type)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    custom_bn_type = os.environ.get('bn_type', 'inplace_abn')

    if int(os.environ.get('eval_os_8', 1)):
        print("Complexity Evaluation Results for PPM with input shape [2048 X 128 X 128]")
        feats = torch.randn((1, 2048, 128, 128)).cuda()
        psp_infer = PSPModule(2048, bn_type=custom_bn_type)
    else:
        print("Complexity Evaluation Results for PPM with input shape [720 X 256 X 512]")
        feats = torch.randn((1, 720, 256, 512)).cuda()
        psp_infer = PSPModule(720, bn_type=custom_bn_type)

    psp_infer.eval()
    psp_infer.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time
    with torch.no_grad():
        for i in range(100):
            start_time = time.time()
            outputs = psp_infer(feats)
            torch.cuda.synchronize()
            avg_time += (time.time() - start_time)
            avg_mem  += (torch.cuda.max_memory_allocated())

    print("Average Parameters : {}".format(count_parameters(psp_infer)))
    print("Average Running Time: {}".format(avg_time/100))
    print("Average GPU Memory: {:.2f} MB".format(avg_mem / 100 / 2**20))
    print("\n\n")