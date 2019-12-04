##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Modified from: https://github.com/AlexHex7/Non-local_pytorch
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import pdb
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from lib.models.tools.module_helper import ModuleHelper

BN_MOMENTUM = 0.1

class ObjectContext_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(ObjectContext_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 

        # probs = F.normalize(probs, p=2, dim=1)
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        cc = torch.matmul(probs, feats)# batch x k x c

        # visualize the region maps
        # region_maps = probs[0].view(c, h, w)
        # from lib.vis.attention_visualizer import visualize_map   
        # visualize_map(region_maps, [h, w],
        #     out_path="/msravcshare/yuyua/code/segmentation/openseg.pytorch/visualize/region_maps/")
        return cc.permute(0, 2, 1).unsqueeze(3)


class ObjectContext_Channel_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(ObjectContext_Channel_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 

        probs = F.softmax(self.scale * probs, dim=1)# batch x k x hw
        cc = torch.matmul(probs, feats)# batch x k x c
        # visualize the region maps
        # region_maps = probs[0].view(c, h, w)
        # from lib.vis.attention_visualizer import visualize_map   
        # visualize_map(region_maps, [h, w],
        #     out_path="/msravcshare/yuyua/code/segmentation/openseg.pytorch/visualize/region_maps/")
        return cc.permute(0, 2, 1).unsqueeze(3)


class ObjectContext_Module_Visualize(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(ObjectContext_Module_Visualize, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs, label):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)              # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw

        ##########################################################
        # label = F.upsample(input=label.unsqueeze(1).type(torch.cuda.FloatTensor), size=(h, w), mode='nearest')
        # label = label[0,:,:,:].view(-1)
        # for k in range(19):
        #     print("OCM: histogram for pixels belonging to category: "+str(k))
        #     mask = (label[:] == k)
        #     for j in range(19):
        #         print(torch.sum(probs[0,j,mask]))
        #     print("*********************************************")
        ##########################################################
                
        cc = torch.matmul(probs, feats)# batch x k x c
        return cc.permute(0, 2, 1).unsqueeze(3)


class Weighted_ObjectContext_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
        # L^2-Norm
        # tensor([927.6089, 481.7609, 955.0940, 168.7788, 157.7509, 
        #         280.6083, 224.3742, 209.2754, 257.4268, 213.7205,
        #         249.2374, 170.6936, 252.7385, 359.6712, 179.1650,
        #         173.6340, 141.2892, 247.5613, 228.1175]

        # max_prob
        # (tensor([0.9998, 0.9991, 0.9997, 0.3641, 0.6793, 
        #          0.9991, 0.8959, 0.9990, 0.9991, 0.4206, 
        #          0.9979, 0.9978, 0.0905, 0.9991, 0.5152, 
        #          0.3482, 0.3752, 0.0591, 0.1194]
    """
    def __init__(self, cls_num=0):
        super(Weighted_ObjectContext_Module, self).__init__()
        self.cls_num = cls_num
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        max_prob, _ = torch.max(F.softmax(probs, dim=1), dim=2)
        max_prob = max_prob.unsqueeze(-1)
        probs = F.softmax(probs, dim=2) # batch x k x hw
        cc = torch.matmul(probs, feats) # batch x k x c
        cc = torch.mul(max_prob, cc)
        return cc.permute(0, 2, 1).unsqueeze(3)


class NonObjectContext_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(NonObjectContext_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(-1 * probs, dim=2)# batch x k x hw
        cc = torch.matmul(probs, feats)# batch x k x c
        return cc.permute(0, 2, 1).unsqueeze(3)


class ObjectContext_Module_double_softmax(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=10):
        super(ObjectContext_Module_double_softmax, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(probs, dim=1)
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        cc = torch.matmul(probs, feats)# batch x k x c
        return cc.permute(0, 2, 1).unsqueeze(3)


class ObjectContext_Module_L2Norm(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(ObjectContext_Module_L2Norm, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.normalize(probs, p=2, dim=1)
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        cc = torch.matmul(probs, feats)# batch x k x c
        return cc.permute(0, 2, 1).unsqueeze(3)


class ObjectContext_Module_Dropout(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(ObjectContext_Module_Dropout, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = F.dropout(probs, 0.1, training=self.training, inplace=False)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        cc = torch.matmul(probs, feats)# batch x k x c
        return cc.permute(0, 2, 1).unsqueeze(3)


class ObjectContext_Module_onehot(nn.Module):
    """
        Convert the intial prediction to one-hot attention maps,
        then we aggregate the context information according to the one-hot attention map.
    """
    def __init__(self, cls_num=0, scale=1):
        super(ObjectContext_Module_onehot, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c

        probs = probs.permute(0, 2, 1) # batch x hw x k
        labels = torch.argmax(probs, dim=2) # batch x hw x 1
        labels = labels.view(-1, 1)
        labels_onehot = torch.zeros(batch_size*h*w, c).cuda()
        labels_onehot = labels_onehot.scatter_(1, labels, 1)
        labels_onehot = labels_onehot.view(batch_size, h*w, -1) # batch x hw x k
        labels_onehot = labels_onehot.permute(0, 2, 1) # batch x k x hw
        labels_onehot = F.normalize(labels_onehot, p=1, dim=2)

        cc = torch.matmul(labels_onehot, feats)# batch x k x c
        return cc.permute(0, 2, 1).unsqueeze(3)


class _ObjectAttentionBlock(nn.Module):
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
    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type, momentum=BN_MOMENTUM),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type, momentum=BN_MOMENTUM),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type, momentum=BN_MOMENTUM),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type, momentum=BN_MOMENTUM),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type, momentum=BN_MOMENTUM),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type, momentum=BN_MOMENTUM),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # visualize the assignment maps
        # assign_map = sim_map[0].permute(1, 0)
        # assign_map = assign_map.view(19, h, w)
        # from lib.vis.attention_visualizer import visualize_map
        # visualize_map(assign_map, [h, w],
        #     out_path="/msravcshare/yuyua/code/segmentation/openseg.pytorch/visualize/assign_maps/")

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                    key_channels,
                                                    scale, bn_type=bn_type)


class FastBaseOC_Module(nn.Module):
    """
    Implementation of the FastOC module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale, dropout=0.1, bn_type=None):
        super(FastBaseOC_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale=1, bn_type=bn_type)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2*in_channels, out_channels, kernel_size=1, padding=0),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type, momentum=BN_MOMENTUM),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class FastBaseOC_Context_Module(nn.Module):
    """
    Implementation of the FastOC module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, in_channels, key_channels, scale, bn_type=None):
        super(FastBaseOC_Context_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale=1, bn_type=bn_type)
        
    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        return context


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    probs = torch.randn((1, 19, 128, 128)).cuda()
    feats = torch.randn((1, 2048, 128, 128)).cuda()


    ocp_infer = ObjectContext_Module(19)
    fastbaseoc_infer = FastBaseOC_Module(in_channels=2048,
                                       key_channels=256, 
                                       out_channels=256, 
                                       scale=1,
                                       dropout=0, 
                                       bn_type='inplace_abn')

    ocp_infer.eval()
    fastbaseoc_infer.eval()
    ocp_infer.cuda()
    fastbaseoc_infer.cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
  
    avg_time = 0
    avg_mem  = 0
    import time
    with torch.no_grad():
        for i in range(100):
            start_time = time.time()
            ocp_feats = ocp_infer(feats, probs)
            outputs = fastbaseoc_infer(feats, ocp_feats)
            torch.cuda.synchronize()
            avg_time += (time.time() - start_time)
            avg_mem  += (torch.cuda.max_memory_allocated()-feats.element_size() * feats.nelement())

    print("Average Parameters : {}".format(count_parameters(fastbaseoc_infer)))
    print("Average Running Time: {}".format(avg_time/100))
    print("Average GPU Memory: {}".format(avg_mem/100))