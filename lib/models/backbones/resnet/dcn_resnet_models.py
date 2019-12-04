#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Deformable ConvNets v2: More Deformable, Better Results
# Modified by: RainbowSecret(yuyua@microsoft.com)
# Select Seg Model for img segmentation.

import pdb
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from collections import OrderedDict

from lib.models.tools.module_helper import ModuleHelper
from lib.extensions.dcn import ModulatedDeformConv, ModulatedDeformRoIPoolingPack, DeformConv 

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 bn_type=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 with_dcn=False,
                 num_deformable_groups=1,
                 dcn_offset_lr_mult=0.1,
                 use_regular_conv_on_stride=False,
                 use_modulated_dcn=False,
                 bn_type=None):
        """Bottleneck block.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        conv1_stride = 1
        conv2_stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=conv1_stride, bias=False)

        self.with_dcn = with_dcn
        self.use_modulated_dcn = use_modulated_dcn
        if use_regular_conv_on_stride and stride > 1:
            self.with_dcn = False
        if self.with_dcn:
            print("--->> use {}dcn in block where c_in={} and c_out={}".format(
                'modulated ' if self.use_modulated_dcn else '', planes, inplanes))
            if use_modulated_dcn:
                self.conv_offset_mask = nn.Conv2d(
                    planes,
                    num_deformable_groups * 27,
                    kernel_size=3,
                    stride=conv2_stride,
                    padding=dilation,
                    dilation=dilation)
                self.conv_offset_mask.lr_mult = dcn_offset_lr_mult
                self.conv_offset_mask.zero_init = True

                self.conv2 = ModulatedDeformConv(planes, planes, 3, stride=conv2_stride,
                                          padding=dilation, dilation=dilation,
                                          deformable_groups=num_deformable_groups, no_bias=True)
            else:
                self.conv2_offset = nn.Conv2d(
                    planes,
                    num_deformable_groups * 18,
                    kernel_size=3,
                    stride=conv2_stride,
                    padding=dilation,
                    dilation=dilation)
                self.conv2_offset.lr_mult = dcn_offset_lr_mult
                self.conv2_offset.zero_init = True

                self.conv2 = DeformConv(planes, planes, (3, 3), stride=conv2_stride,
                    padding=dilation, dilation=dilation,
                    num_deformable_groups=num_deformable_groups)
        else:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)


        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            if self.with_dcn:
                if self.use_modulated_dcn:
                    offset_mask = self.conv_offset_mask(out)
                    offset1, offset2, mask_raw = torch.chunk(offset_mask, 3, dim=1)
                    offset = torch.cat((offset1, offset2), dim=1)
                    mask = torch.sigmoid(mask_raw)
                    out = self.conv2(out, offset, mask)
                else:
                    offset = self.conv2_offset(out)
                    # add bias to the offset to solve the bug of dilation rates within dcn.
                    dilation = self.conv2.dilation[0]
                    bias_w = torch.cuda.FloatTensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) * (dilation - 1)
                    bias_h = bias_w.permute(1, 0)
                    bias_w.requires_grad = False
                    bias_h.requires_grad = False
                    offset += torch.cat([bias_h.reshape(-1), bias_w.reshape(-1)]).view(1, -1, 1, 1)
                    out = self.conv2(out, offset)
            else:
                out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out = out + residual
            return out


        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu_in(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   with_dcn=False,
                   dcn_offset_lr_mult=0.1,
                   use_regular_conv_on_stride=False,
                   use_modulated_dcn=False,
                   bn_type=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp,
            with_dcn=with_dcn,
            dcn_offset_lr_mult=dcn_offset_lr_mult,
            use_regular_conv_on_stride=use_regular_conv_on_stride,
            use_modulated_dcn=use_modulated_dcn,
            bn_type=bn_type))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp, with_dcn=with_dcn, 
                  dcn_offset_lr_mult=dcn_offset_lr_mult, use_regular_conv_on_stride=use_regular_conv_on_stride,
                  use_modulated_dcn=use_modulated_dcn, bn_type=bn_type))

    return nn.Sequential(*layers)


class DCNResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """
    def __init__(self,
                 block,
                 layers,
                 deep_base=True,
                 bn_type=None):
        super(DCNResNet, self).__init__()
        # if depth not in self.arch_settings:
        #     raise KeyError('invalid depth {} for resnet'.format(depth))
        # assert num_stages >= 1 and num_stages <= 4
        # block, stage_blocks = self.arch_settings[depth]
        # stage_blocks = stage_blocks[:num_stages]
        # assert len(strides) == len(dilations) == num_stages
        # assert max(out_indices) < num_stages
        self.style = 'pytorch'
        self.inplanes = 128 if deep_base else 64
        if deep_base:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)),
                ('bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)),
                ('relu1', nn.ReLU(inplace=False)),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)),
                ('relu2', nn.ReLU(inplace=False)),
                ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.inplanes)),
                ('relu3', nn.ReLU(inplace=False))]
            ))
        else:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.inplanes)),
                ('relu1', nn.ReLU(inplace=False))]
            ))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = make_res_layer(
                block,
                self.inplanes,
                64,
                layers[0],
                style=self.style,
                with_dcn=False,
                use_modulated_dcn=False,
                bn_type=bn_type)

        self.layer2 = make_res_layer(
                block,
                256,
                128,
                layers[1],
                stride=2,
                style=self.style,
                with_dcn=False,
                use_modulated_dcn=False,
                bn_type=bn_type)

        self.layer3 = make_res_layer(
                block,
                512,
                256,
                layers[2],
                stride=2,
                style=self.style,
                with_dcn=True,
                use_modulated_dcn=False,
                bn_type=bn_type)

        self.layer4 = make_res_layer(
                block,
                1024,
                512,
                layers[3],
                stride=2,
                style=self.style,
                with_dcn=True,
                use_modulated_dcn=False,
                bn_type=bn_type)


    def forward(self, x):
        x = self.resinit(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x



class DCNResNetModels(object):

    def __init__(self, configer):
        self.configer = configer

    def deepbase_dcn_resnet50(self, **kwargs):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = DCNResNet(Bottleneck, [3, 4, 6, 3], deep_base=True,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, 
                                        all_match=False, 
                                        pretrained=self.configer.get('network', 'pretrained'),
                                        network="dcnet")
        return model

    def deepbase_dcn_resnet101(self, **kwargs):
        """Constructs a ResNet-101 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = DCNResNet(Bottleneck, [3, 4, 23, 3], deep_base=True,
                       bn_type=self.configer.get('network', 'bn_type'), **kwargs)
        model = ModuleHelper.load_model(model, 
                                        all_match=False, 
                                        pretrained=self.configer.get('network', 'pretrained'),
                                        network="dcnet")
        return model