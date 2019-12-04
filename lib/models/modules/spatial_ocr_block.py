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
import pdb
import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper


def label_to_onehot(gt, num_classes, ignore_index=-1):
    '''
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    '''
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)          

    return onehot.permute(0, 3, 1, 2)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1, use_gt=False):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.use_gt = use_gt
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs, gt_probs=None):
        if self.use_gt and gt_probs is not None:
            gt_probs = label_to_onehot(gt_probs.squeeze(1).type(torch.cuda.LongTensor), probs.size(1))
            batch_size, c, h, w = gt_probs.size(0), gt_probs.size(1), gt_probs.size(2), gt_probs.size(3)
            gt_probs = gt_probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1) # batch x hw x c 
            gt_probs = F.normalize(gt_probs, p=1, dim=2)# batch x k x hw
            ocr_context = torch.matmul(gt_probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
            return ocr_context               
        else:
            batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
            probs = probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1) # batch x hw x c 
            probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
            ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
            return ocr_context


class PyramidSpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scales=[1, 2, 4]):
        super(PyramidSpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scales = scales
        self.relu = nn.ReLU(inplace=True)

    def _compute_single_scale(self, feats, probs, dh, dw):
        batch_size, k, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        c = feats.size(1)

        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:              # padding in both left&right sides
            feats = F.pad(feats, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
            probs = F.pad(probs, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))

        feats = feats.view(batch_size, c, out_h, dh, out_w, dw).permute(0, 3, 5, 1, 2, 4)
        feats = feats.contiguous().view(batch_size, dh*dw, c, out_h, out_w)

        probs = probs.view(batch_size, k, out_h, dh, out_w, dw).permute(0, 3, 5, 1, 2, 4)
        probs = probs.contiguous().view(batch_size, dh*dw, k, out_h, out_w)

        feats = feats.view(batch_size, dh*dw, c, -1)
        probs = probs.view(batch_size, dh*dw, k, -1)
        feats = feats.permute(0, 1, 3, 2)

        probs = F.softmax(probs, dim=3)# batch x k x hw
        cc = torch.matmul(probs, feats).view(batch_size, -1, c)# batch x k x c

        return cc.permute(0, 2, 1).unsqueeze(3)

    def forward(self, feats, probs):
        ocr_list = []
        for scale in self.scales:
            ocr_tmp = self._compute_single_scale(feats, probs, scale, scale)
            ocr_list.append(ocr_tmp)
        pyramid_ocr= torch.cat(ocr_list, 2)
        return pyramid_ocr


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(1).type(torch.cuda.LongTensor), proxy.size(2)-1)
            sim_map = gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size, h*w, -1)
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value) # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(input=bg_context, size=(h, w), mode='bilinear', align_corners=True)
            return context, bg_context
        else:
            if self.fetch_attention:
                return context, sim_map
            else:
                return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 use_gt=False, 
                 use_bg=False,
                 fetch_attention=False, 
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     use_gt,
                                                     use_bg,
                                                     fetch_attention,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 use_gt=False,
                 use_bg=False,
                 use_oc=True,
                 fetch_attention=False, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           use_gt,
                                                           use_bg,
                                                           fetch_attention,
                                                           bn_type)
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats, proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats, gt_label)
        else:
            if self.fetch_attention:
                context, sim_map = self.object_context_block(feats, proxy_feats)
            else:
                context = self.object_context_block(feats, proxy_feats)

        if self.use_bg:
            if self.use_oc:
                output = self.conv_bn_dropout(torch.cat([context, bg_context, feats], 1))
            else:
                output = self.conv_bn_dropout(torch.cat([bg_context, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        if self.fetch_attention:
            return output, sim_map
        else:
            return output


class SpatialOCR_Context(nn.Module):
    """
    Implementation of the FastOC module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, in_channels, key_channels, scale=1, dropout=0, bn_type=None,):
        super(SpatialOCR_Context, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type=bn_type)
        
    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        return context



class SpatialOCR_ASP_Module(nn.Module):
    def __init__(self, features, hidden_features=256, out_features=512, dilations=(12, 24, 36), num_classes=19, bn_type=None, dropout=0.1):
        super(SpatialOCR_ASP_Module, self).__init__()
        from lib.models.modules.spatial_ocr_block import SpatialOCR_Context
        self.context = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),
                                     SpatialOCR_Context(in_channels=hidden_features,
                                                        key_channels=hidden_features//2, scale=1, bn_type=bn_type),
                                    )
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, padding=0, dilation=1, bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv5 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=True),
                                   ModuleHelper.BNReLU(hidden_features, bn_type=bn_type),)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=True),
            ModuleHelper.BNReLU(out_features, bn_type=bn_type),
            nn.Dropout2d(dropout)
            )
        self.object_head = SpatialGather_Module(num_classes)

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5):
        assert(len(feat1)==len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x, probs):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context[0](x)
        feat1 = self.context[1](feat1)
        proxy_feats = self.object_head(feat1, probs)
        feat1 = self.context[2](feat1, proxy_feats)
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
    probs = torch.randn((1, 19, 128, 128)).cuda()
    feats = torch.randn((1, 2048, 128, 128)).cuda()

    conv_3x3 = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
        ModuleHelper.BNReLU(512, bn_type='inplace_abn'),
    )

    ocp_gather_infer = SpatialGather_Module(19)
    ocp_distr_infer = SpatialOCR_Module(in_channels=512,
                                         key_channels=256, 
                                         out_channels=512,
                                         scale=1,
                                         dropout=0, 
                                         bn_type='inplace_abn')

    ocp_gather_infer.eval()
    ocp_distr_infer.eval()
    conv_3x3.eval()
    ocp_gather_infer.cuda()
    ocp_distr_infer.cuda()
    conv_3x3.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
  
    avg_time = 0
    avg_mem  = 0
    import time
    with torch.no_grad():
        for i in range(100):
            start_time = time.time()
            feats_ = conv_3x3(feats)
            ocp_feats = ocp_gather_infer(feats_, probs)
            outputs = ocp_distr_infer(feats_, ocp_feats)
            torch.cuda.synchronize()
            avg_time += (time.time() - start_time)
            avg_mem  += (torch.cuda.max_memory_allocated()-feats.element_size() * feats.nelement())

    print("Average Parameters : {}".format(count_parameters(ocp_distr_infer)+count_parameters(conv_3x3)))
    print("Average Running Time: {}".format(avg_time/100))
    print("Average GPU Memory: {}".format(avg_mem/100))