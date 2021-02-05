##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.models.modules.base_oc_block import BaseOC_Context_Module
from lib.models.tools.module_helper import ModuleHelper


class ASP_OC_Module(nn.Module):
    def __init__(self, features, out_features=256, dilations=(12, 24, 36), bn_type=None, dropout=0.1):
        super(ASP_OC_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(out_features, bn_type=bn_type),
                                     BaseOC_Context_Module(in_channels=out_features, out_channels=out_features,
                                                              key_channels=out_features//2, value_channels=out_features//2,
                                                              dropout=0, sizes=([2]), bn_type=bn_type))
        self.conv2 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv3 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv4 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type))
        self.conv5 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   ModuleHelper.BNReLU(out_features, bn_type=bn_type))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features * 2, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_features * 2, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    feats = torch.randn((1, 2048, 128, 128)).cuda()

    conv_3x3 = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
        ModuleHelper.BNReLU(512, bn_type='torchsyncbn'),
    )
    aspoc_infer = ASP_OC_Module(512,
                                256,
                                bn_type='torchsyncbn')

    aspoc_infer.eval()
    conv_3x3.eval()
    aspoc_infer.cuda()
    conv_3x3.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time
    # with torch.no_grad()
    for i in range(100):
        start_time = time.time()
        outputs = conv_3x3(feats)
        outputs = aspoc_infer(outputs)
        torch.cuda.synchronize()
        avg_time += (time.time() - start_time)
        avg_mem  += (torch.cuda.memory_allocated()-feats.element_size() * feats.nelement())

    print("Average Parameters : {}".format(count_parameters(aspoc_infer)+count_parameters(conv_3x3)))
    print("Average Running Time: {}".format(avg_time/100))
    print("Average GPU Memory: {}".format(avg_mem/100))