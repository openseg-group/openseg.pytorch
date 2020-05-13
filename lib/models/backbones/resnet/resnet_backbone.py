#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import torch
import torch.nn as nn

from lib.models.backbones.resnet.resnet_models import ResNetModels
from lib.models.backbones.resnet.resnext_models import ResNextModels
from lib.models.backbones.resnet.resnest_models import ResNeStModels

# if torch.__version__[:3] == '0.4':
#     from lib.models.backbones.resnet.dcn_resnet_models import DCNResNetModels

class NormalResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()

        self.num_features = 2048
        # take pretrained resnet, except AvgPool and FC
        self.resinit = orig_resnet.resinit
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.resinit(x)
        tuple_features.append(x)
        x = self.maxpool(x)
        tuple_features.append(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features


class DilatedResnetBackbone(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, multi_grid=(1, 2, 4)):
        super(DilatedResnetBackbone, self).__init__()

        self.num_features = 2048
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(4 * r)))

        elif dilate_scale == 16:
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(2 * r)))

        # Take pretrained resnet, except AvgPool and FC
        self.resinit = orig_resnet.resinit
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.resinit(x)
        tuple_features.append(x)
        x = self.maxpool(x)
        tuple_features.append(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features


class ResNetBackbone(object):
    def __init__(self, configer):
        self.configer = configer
        self.resnet_models = ResNetModels(self.configer)
        self.resnext_models = ResNextModels(self.configer)
        self.resnest_models = ResNeStModels(self.configer)

        # if torch.__version__[:3] == '0.4':
        #     self.dcn_resnet_models = DCNResNetModels(self.configer)

    def __call__(self):
        arch = self.configer.get('network', 'backbone')
        multi_grid = None
        if self.configer.exists('network', 'multi_grid'):
            multi_grid = self.configer.get('network', 'multi_grid')

        if arch == 'deepbase_resnet18':
            orig_resnet = self.resnet_models.deepbase_resnet18()
            arch_net = NormalResnetBackbone(orig_resnet)
            arch_net.num_features = 512

        elif arch == 'deepbase_resnet18_dilated8':
            orig_resnet = self.resnet_models.deepbase_resnet18()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)
            arch_net.num_features = 512

        elif arch == 'deepbase_resnet18_dilated16':
            orig_resnet = self.resnet_models.deepbase_resnet18()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)
            arch_net.num_features = 512

        elif arch == 'resnet34':
            orig_resnet = self.resnet_models.resnet34()
            arch_net = NormalResnetBackbone(orig_resnet)
            arch_net.num_features = 512

        elif arch == 'resnet34_dilated8':
            orig_resnet = self.resnet_models.resnet34()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)
            arch_net.num_features = 512

        elif arch == 'resnet34_dilated16':
            orig_resnet = self.resnet_models.resnet34()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)
            arch_net.num_features = 512

        elif arch == 'resnet50':
            orig_resnet = self.resnet_models.resnet50()
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'resnet50_dilated8':
            orig_resnet = self.resnet_models.resnet50()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif arch == 'resnet50_dilated16':
            orig_resnet = self.resnet_models.resnet50()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        elif arch == 'deepbase_resnet50':
            orig_resnet = self.resnet_models.deepbase_resnet50()
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'deepbase_resnet50_dilated8':
            orig_resnet = self.resnet_models.deepbase_resnet50()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif arch == 'deepbase_resnet50_dilated16':
            orig_resnet = self.resnet_models.deepbase_resnet50()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        elif arch == 'resnet101':
            orig_resnet = self.resnet_models.resnet101()
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'resnet101_dilated8':
            orig_resnet = self.resnet_models.resnet101()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif arch == 'resnet101_dilated16':
            orig_resnet = self.resnet_models.resnet101()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        elif arch == 'deepbase_resnet101':
            orig_resnet = self.resnet_models.deepbase_resnet101()
            arch_net = NormalResnetBackbone(orig_resnet)

        elif arch == 'deepbase_resnet101_dilated8':
            orig_resnet = self.resnet_models.deepbase_resnet101()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif arch == 'deepbase_resnet101_dilated16':
            orig_resnet = self.resnet_models.deepbase_resnet101()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        elif arch == 'deepbase_resnet152_dilated8':
            orig_resnet = self.resnet_models.deepbase_resnet152()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif arch == 'deepbase_resnet152_dilated16':
            orig_resnet = self.resnet_models.deepbase_resnet152()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=16, multi_grid=multi_grid)

        # resnext models
        elif arch == 'resnext101_32x8d_dilated8':
            orig_resnet = self.resnext_models.resnext101_32x8d()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif arch == 'resnext101_32x16d_dilated8':
            orig_resnet = self.resnext_models.resnext101_32x16d()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif arch == 'resnext101_32x32d_dilated8':
            orig_resnet = self.resnext_models.resnext101_32x32d()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        elif arch == 'resnext101_32x48d_dilated8':
            orig_resnet = self.resnext_models.resnext101_32x48d()
            arch_net = DilatedResnetBackbone(orig_resnet, dilate_scale=8, multi_grid=multi_grid)

        # deformable resnet models
        # elif arch == 'deepbase_dcn_resnet50_dilated8':
        #     if torch.__version__[:3] != '0.4':
        #         raise NotImplementedError
        #     orig_dcn_resnet = self.dcn_resnet_models.deepbase_dcn_resnet50()
        #     arch_net = DilatedResnetBackbone(orig_dcn_resnet, dilate_scale=8, multi_grid=multi_grid)

        # elif arch == 'deepbase_dcn_resnet50_dilated16':
        #     if torch.__version__[:3] != '0.4':
        #         raise NotImplementedError
        #     orig_dcn_resnet = self.dcn_resnet_models.deepbase_dcn_resnet50()
        #     arch_net = DilatedResnetBackbone(orig_dcn_resnet, dilate_scale=16, multi_grid=multi_grid)

        # elif arch == 'deepbase_dcn_resnet101_dilated8':
        #     if torch.__version__[:3] != '0.4':
        #         raise NotImplementedError
        #     orig_dcn_resnet = self.dcn_resnet_models.deepbase_dcn_resnet101()
        #     arch_net = DilatedResnetBackbone(orig_dcn_resnet, dilate_scale=8, multi_grid=multi_grid)

        # elif arch == 'deepbase_dcn_resnet101_dilated16':
        #     if torch.__version__[:3] != '0.4':
        #         raise NotImplementedError
        #     orig_dcn_resnet = self.dcn_resnet_models.deepbase_dcn_resnet101()
        #     arch_net = DilatedResnetBackbone(orig_dcn_resnet, dilate_scale=16, multi_grid=multi_grid)

        elif arch == 'wide_resnet16_dilated8':
            arch_net = self.resnet_models.wide_resnet16()

        elif arch == 'wide_resnet20_dilated8':
            arch_net = self.resnet_models.wide_resnet20()

        elif arch == 'wide_resnet38_dilated8':
            arch_net = self.resnet_models.wide_resnet38()

        # ResNeSt series: https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/resnest.py
        elif arch == 'deepbase_resnest50_dilated8':
            arch_net = self.resnest_models.deepbase_resnest50()

        elif arch == 'deepbase_resnest101_dilated8':
            arch_net = self.resnest_models.deepbase_resnest101()

        elif arch == 'deepbase_resnest200_dilated8':
            arch_net = self.resnest_models.deepbase_resnest200()

        elif arch == 'deepbase_resnest269_dilated8':
            arch_net = self.resnest_models.deepbase_resnest269()

        else:
            raise Exception('Architecture undefined!')

        return arch_net
