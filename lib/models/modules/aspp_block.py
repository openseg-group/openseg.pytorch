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

from lib.models.tools.module_helper import ModuleHelper
from functools import partial


class ASPPModuleV2(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. 
        *"Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs."*
    """
    def __init__(self, features, inner_features=512, out_features=512, dilations=(12, 24, 36), bn_type=None, dropout=0.1):
        super(ASPPModuleV2, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv_3x3_1 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv_3x3_2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv_3x3_3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.fuse = nn.Sequential(
            nn.Conv2d(inner_features * 4, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = self.conv_1x1(x)
        feat2 = self.conv_3x3_1(x)
        feat3 = self.conv_3x3_2(x)
        feat4 = self.conv_3x3_3(x)
        out = torch.cat((feat1, feat2, feat3, feat4), 1)
        out = self.fuse(out)
        return out

class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=256, dilations=(12, 24, 36), bn_type=None, dropout=0.1):
        super(ASPPModule, self).__init__()

        self.conv_gp = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   ModuleHelper.BNReLU(inner_features, bn_type=bn_type))

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv_3x3_1 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv_3x3_2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))
        self.conv_3x3_3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            ModuleHelper.BNReLU(inner_features, bn_type=bn_type))

        self.fuse = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat_gp = F.interpolate(self.conv_gp(x), size=(h, w), mode='bilinear', align_corners=True)
        feat1 = self.conv_1x1(x)
        feat2 = self.conv_3x3_1(x)
        feat3 = self.conv_3x3_2(x)
        feat4 = self.conv_3x3_3(x)
        out = torch.cat((feat_gp, feat1, feat2, feat3, feat4), 1)
        out = self.fuse(out)
        return out


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    custom_bn_type = os.environ.get('bn_type', 'inplace_abn')

    if int(os.environ.get('eval_os_8', 1)):
        print("Complexity Evaluation Results for ASPP with input shape [2048 X 128 X 128]")
        feats = torch.randn((1, 2048, 128, 128)).cuda()
        aspp_infer = ASPPModule(2048, 256, 256, bn_type=custom_bn_type)
    else:
        print("Complexity Evaluation Results for ASPP with input shape [720 X 256 X 512]")
        feats = torch.randn((1, 720, 256, 512)).cuda()
        aspp_infer = ASPPModule(720, 256, 256, bn_type=custom_bn_type)

    aspp_infer.eval()
    aspp_infer.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time
    with torch.no_grad():
        for i in range(100):
            start_time = time.time()
            outputs = aspp_infer(feats)
            torch.cuda.synchronize()
            avg_time += (time.time() - start_time)
            avg_mem  += (torch.cuda.max_memory_allocated()-feats.element_size() * feats.nelement())

    print("Average Parameters : {}".format(count_parameters(aspp_infer)))
    print("Average Running Time: {}".format(avg_time/100))
    print("Average GPU Memory: {:.2f} MB".format(avg_mem / 100 / 2**20))
    print("\n\n")