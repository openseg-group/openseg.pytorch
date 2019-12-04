##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret, HuangLang
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import math
import pdb
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

from lib.models.tools.module_helper import ModuleHelper


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, bn_type=None):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )

        self.f_value = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                kernel_size=1, stride=1, padding=0)
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, 
                kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, bn_type=None):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                    key_channels,
                                                    value_channels,
                                                    out_channels,
                                                    scale, bn_type)


class LongShortSelfAttention(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factor=[8,8], bn_type=None):
        super(LongShortSelfAttention, self).__init__()
        self.out_channels = out_channels
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.down_factor = down_factor
        self.long_range_sa = SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
        self.short_range_sa = SelfAttentionBlock2D(out_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
    
    def forward(self, x):
        n, c, h, w = x.size()
        dh, dw = self.down_factor       # down_factor for h and w, respectively
        
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        # maybe do some padding
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:              # padding in both left&right sides
            feats = F.pad(x, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
        else:
            feats = x
        
        # pixel unshuffle feature map
        feats = feats.view(n, c, out_h, dh, out_w, dw)
        feats = feats.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)

        # do long range self-attention
        feats = self.long_range_sa(feats)
        c = self.out_channels

        # pixel shuffle feature map
        feats = feats.view(n, dh, dw, c, out_h, out_w)

        # slice feature map into grids
        feats = feats.permute(0, 4, 5, 3, 1, 2).contiguous().view(-1, c, dh, dw)

        # do short range self-attention
        feats = self.short_range_sa(feats)
        
        # concat sliced feature map
        feats = feats.view(n, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5)
        feats = feats.contiguous().view(n, c, dh * out_h, dw * out_w)

        # maybe remove padding
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w]
        
        return feats


class ISA_Module(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factors=[[8,8]], dropout=0, bn_type=None):
        super(ISA_Module, self).__init__()

        assert isinstance(down_factors, (tuple, list))
        self.down_factors = down_factors

        self.stages = [self._make_stage(in_channels, key_channels, value_channels, out_channels, d, bn_type) for d in down_factors]
        self.stages = nn.ModuleList(self.stages)

        concat_channels = in_channels + out_channels
        if len(self.down_factors) > 1:
            self.up_conv = nn.Sequential(
                nn.Conv2d(in_channels, len(self.down_factors) * out_channels, kernel_size=1, padding=0),
                ModuleHelper.BNReLU(len(self.down_factors) * out_channels, bn_type=bn_type),
            )
            concat_channels = out_channels * len(self.down_factors) * 2
        
        self.conv_bn = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, padding=0),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout),
        )
    

    def _make_stage(self, in_channels, key_channels, value_channels, out_channels, down_factor, bn_type):
        return LongShortSelfAttention(in_channels, key_channels, value_channels, out_channels, down_factor, bn_type)
    
    def forward(self, x):
        priors = [stage(x) for stage in self.stages]
        if len(self.down_factors) == 1:
            context = priors[0]
        else:
            context = torch.cat(priors, dim=1)
            x = self.up_conv(x)
        
        # residual connection
        return self.conv_bn(torch.cat([x, context], dim=1))

        
class LongRangeSelfAttention(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factor=[8, 8], bn_type=None):
        super(LongRangeSelfAttention, self).__init__()
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.down_factor = down_factor
        self.out_channels = out_channels
        self.sa = SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
    
    def forward(self, x):
        n, c, h, w = x.size()
        dh, dw = self.down_factor       # down_factor for h and w, respectively
        
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        
        # padding and reshape: [n x c x h x w] ===> [(nxdhxdw) x c x out_h x out_w]
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:              # center padding
            feats = F.pad(x, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
        else:
            feats = x
        feats = feats.view(n, c, out_h, dh, out_w, dw)
        feats = feats.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)

        # apply self-attention on nxdhxdw different sparsely-sampled regions.
        feats = self.sa(feats)
        c = self.out_channels

        # pixel shuffle
        # reshape and de-padding : [(nxdhxdw) x c x out_h x out_w] ===> [n x c x h x w]
        feats = feats.view(n, dh, dw, c, out_h, out_w).permute(0, 3, 4, 1, 5, 2)
        feats = feats.contiguous().view(n, c, out_h * dh, out_w * dw)
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w]
        
        return feats


class ShortRangeSelfAttention(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factor=[8, 8], bn_type=None):
        super(ShortRangeSelfAttention, self).__init__()
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.out_channels = out_channels
        self.down_factor = down_factor
        self.sa = SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
    
    def forward(self, x):
        n, c, h, w = x.size()
        dh, dw = self.down_factor           # down_factor for h and w, respectively
        out_h, out_w = h // dh, w // dw
        
        # slice feature map into grids
        #  [n x c x h x w] ===> [(nxout_hxout_w) x c x dh x dw]
        feats = x.view(-1, c, out_h, dh, out_w, dw).permute(0, 2, 4, 1, 3, 5)
        feats = feats.contiguous().view(-1, c, dh, dw)

        # do short range self-attention
        feats = self.sa(feats)
        c = self.out_channels

        # concat sliced feature map
        # [n x out_h x out_w x c x dh x dw] ===> [n x c x out_h x dh x out_w x dw]
        feats = feats.view(n, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5)
        feats = feats.contiguous().view(n, c, out_h * dh, out_w * dw)
        return feats


class ISA_LONG_Module(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factors=(4,), dropout=0, bn_type=None):
        super(ISA_LONG_Module, self).__init__()

        assert isinstance(down_factors, (tuple, list))
        self.down_factors = down_factors

        self.stages = [self._make_stage(in_channels, key_channels, value_channels, out_channels, d, bn_type) for d in down_factors]
        self.stages = nn.ModuleList(self.stages)
        
        concat_channels = in_channels + out_channels
        if len(self.down_factors) > 1:
            self.up_conv = nn.Sequential(
                nn.Conv2d(in_channels, len(self.down_factors) * out_channels, kernel_size=1, padding=0),
                ModuleHelper.BNReLU(len(self.down_factors) * out_channels, bn_type=bn_type),
            )
            concat_channels = out_channels * len(self.down_factors) * 2
        
        self.conv_bn = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, padding=0),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout),
        )
    

    def _make_stage(self, in_channels, key_channels, value_channels, out_channels, down_factor, bn_type):
        return LongRangeSelfAttention(in_channels, key_channels, value_channels, out_channels, down_factor, bn_type)
    
    def forward(self, x):
        priors = [stage(x) for stage in self.stages]

        if len(self.down_factors) == 1:
            context = priors[0]
        else:
            context = torch.cat(priors, dim=1)
            x = self.up_conv(x)
        return self.conv_bn(torch.cat([x, context], dim=1))


class ISA_SHORT_Module(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factors=(4,), dropout=0, bn_type=None):
        super(ISA_SHORT_Module, self).__init__()

        assert isinstance(down_factors, (tuple, list))
        self.down_factors = down_factors

        self.stages = [self._make_stage(in_channels, key_channels, value_channels, out_channels, d, bn_type) for d in down_factors]
        self.stages = nn.ModuleList(self.stages)
        
        concat_channels = in_channels + out_channels
        if len(self.down_factors) > 1:
            self.up_conv = nn.Sequential(
                nn.Conv2d(in_channels, len(self.down_factors) * out_channels, kernel_size=1, padding=0),
                ModuleHelper.BNReLU(len(self.down_factors) * out_channels, bn_type=bn_type),
            )
            concat_channels = out_channels * len(self.down_factors) * 2
        
        self.conv_bn = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, padding=0),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout),
        )
    

    def _make_stage(self, in_channels, key_channels, value_channels, out_channels, down_factor, bn_type):
        return ShortRangeSelfAttention(in_channels, key_channels, value_channels, out_channels, down_factor, bn_type)
    
    def forward(self, x):
        priors = [stage(x) for stage in self.stages]

        if len(self.down_factors) == 1:
            context = priors[0]
        else:
            context = torch.cat(priors, dim=1)
            x = self.up_conv(x)
        return self.conv_bn(torch.cat([x, context], dim=1))


class ASP_ISA_Module(nn.Module):
    def __init__(self, features, out_features=256, dilations=(12, 24, 36), down_factor=[8, 8], bn_type=None, dropout=0.1):
        super(ASP_ISA_Module, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(out_features, bn_type=bn_type),
                                     LongShortSelfAttention(out_features, out_features//2, out_features, out_features, 
                                                down_factor=down_factor, bn_type=bn_type))
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
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    feats = torch.randn((1, 512, 256, 128)).cuda()
    mem = torch.cuda.max_memory_allocated()
    baseoc_infer = ISA_Module(in_channels=512,
                                 key_channels=256,
                                 value_channels=512,
                                 out_channels=512,
                                 dropout=0,
                                 bn_type='inplace_abn')
    baseoc_infer.eval()
    baseoc_infer.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time
    with torch.no_grad():
        for i in range(110):
            start_time = time.time()
            outputs = baseoc_infer(feats)
            torch.cuda.synchronize()
            if i >= 10:
                avg_time += (time.time() - start_time)
                avg_mem += (torch.cuda.max_memory_allocated() - mem)

    print("Average Parameters : {}".format(count_parameters(baseoc_infer)))
    print("Average Running Time: {}".format(avg_time/100))
    print("Average GPU Memory: {}".format(avg_mem/100))