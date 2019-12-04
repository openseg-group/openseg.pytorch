##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Jianyuan Guo, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class CE2P_ASPOCR(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(CE2P_ASPOCR, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        from lib.models.modules.edge_block import Edge_Module
        from lib.models.modules.decoder_block import CE2P_Decoder_Module
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get('network', 'bn_type'), factor=2)
            self.decoder = CE2P_Decoder_Module(self.num_classes, 
                              dropout=0.1, 
                              bn_type=self.configer.get('network', 'bn_type'),
                              inplane1=512,
                              inplane2=512)
        else:
            in_channels = [1024, 2048]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get('network', 'bn_type'), factor=1)
            self.decoder = CE2P_Decoder_Module(self.num_classes, 
                              dropout=0.1, 
                              bn_type=self.configer.get('network', 'bn_type'),
                              inplane1=512,
                              inplane2=256)

        # extra added layers
        from lib.models.modules.spatial_ocr_block import SpatialOCR_ASP_Module
        self.asp_ocr_head = SpatialOCR_ASP_Module(features=2048, 
                                                  hidden_features=256, 
                                                  out_features=512,
                                                  dilations=(6, 12, 18),
                                                  num_classes=self.num_classes,
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.cls = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )

        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_) # x: list output from conv2_x, conv3_x, conv4_x, conv5_x
        seg_dsn = self.dsn(x[-2])
        edge_out, edge_fea = self.edgelayer(x[-4], x[-3], x[-2])
        x5 = x[-1]
        x_hr = self.asp_ocr_head(x5, seg_dsn)
        seg_out1, x_hr = self.decoder(x_hr, x[-4])
        x_hr = torch.cat([x_hr, edge_fea], dim=1)
        seg_out2 = self.cls(x_hr)

        seg_dsn = F.interpolate(seg_dsn, 
                              size=(x_.size(2), x_.size(3)), 
                              mode="bilinear", 
                              align_corners=True)
        seg_out2 = F.interpolate(seg_out2, 
                             size=(x_.size(2), x_.size(3)), 
                             mode="bilinear", 
                             align_corners=True)
        seg_out1 = F.interpolate(seg_out1, 
                             size=(x_.size(2), x_.size(3)), 
                             mode="bilinear", 
                             align_corners=True)
        edge_out = F.interpolate(edge_out, 
                                 size=(x_.size(2), x_.size(3)), 
                                 mode="bilinear", 
                                 align_corners=True)

        return seg_out1, edge_out, seg_dsn, seg_out2


class CE2P_OCRNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(CE2P_OCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        from lib.models.modules.edge_block import Edge_Module
        from lib.models.modules.decoder_block import Decoder_Module
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get('network', 'bn_type'), factor=2)
            self.decoder = Decoder_Module(self.num_classes, 
                              dropout=0.1, 
                              bn_type=self.configer.get('network', 'bn_type'),
                              inplane1=512,
                              inplane2=512)
        else:
            in_channels = [1024, 2048]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get('network', 'bn_type'), factor=1)
            self.decoder = Decoder_Module(self.num_classes, 
                              dropout=0.1, 
                              bn_type=self.configer.get('network', 'bn_type'),
                              inplane1=512,
                              inplane2=256)

        # extra added layers
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module
        self.spatial_context_head = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=2048, 
                                                  key_channels=256, 
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0, 
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.cls = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )

        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_) # x: list output from conv2_x, conv3_x, conv4_x, conv5_x
        seg_dsn = self.dsn(x[-2])
        edge_out, edge_fea = self.edgelayer(x[-4], x[-3], x[-2])
        x5 = x[-1]
        context = self.spatial_context_head(x5, seg_dsn)
        x_hr = self.spatial_ocr_head(x5, context)
        seg_out1, x_hr = self.decoder(x_hr, x[-4])
        x_hr = torch.cat([x_hr, edge_fea], dim=1)
        seg_out2 = self.cls(x_hr)

        seg_dsn = F.interpolate(seg_dsn, 
                              size=(x_.size(2), x_.size(3)), 
                              mode="bilinear", 
                              align_corners=True)
        seg_out2 = F.interpolate(seg_out2, 
                             size=(x_.size(2), x_.size(3)), 
                             mode="bilinear", 
                             align_corners=True)
        seg_out1 = F.interpolate(seg_out1, 
                             size=(x_.size(2), x_.size(3)), 
                             mode="bilinear", 
                             align_corners=True)
        edge_out = F.interpolate(edge_out, 
                                 size=(x_.size(2), x_.size(3)), 
                                 mode="bilinear", 
                                 align_corners=True)

        return seg_out1, edge_out, seg_dsn, seg_out2


class CE2P_IdealOCRNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(CE2P_IdealOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        from lib.models.modules.edge_block import Edge_Module
        from lib.models.modules.decoder_block import Decoder_Module
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get('network', 'bn_type'), factor=2)
            self.decoder = Decoder_Module(self.num_classes, 
                              dropout=0.1, 
                              bn_type=self.configer.get('network', 'bn_type'),
                              inplane1=512,
                              inplane2=512)
        else:
            in_channels = [1024, 2048]
            self.edgelayer = Edge_Module(256, 2, bn_type=self.configer.get('network', 'bn_type'), factor=1)
            self.decoder = Decoder_Module(self.num_classes, 
                              dropout=0.1, 
                              bn_type=self.configer.get('network', 'bn_type'),
                              inplane1=512,
                              inplane2=256)

        # extra added layers
        from lib.models.modules.spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module
        self.spatial_context_head = SpatialGather_Module(self.num_classes, use_gt=True)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=2048, 
                                                  key_channels=256, 
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0, 
                                                  use_gt=True,
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.cls = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            ModuleHelper.BNReLU(256, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )

        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1, bias=False),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_, label_):
        x = self.backbone(x_) # x: list output from conv2_x, conv3_x, conv4_x, conv5_x
        seg_dsn = self.dsn(x[-2])
        edge_out, edge_fea = self.edgelayer(x[-4], x[-3], x[-2])
        x5 = x[-1]

        label = F.interpolate(input=label_.unsqueeze(1).type(torch.cuda.FloatTensor), size=(x5.size(2), x5.size(3)), mode="nearest")
        context = self.spatial_context_head(x5, seg_dsn, label)
        x_hr = self.spatial_ocr_head(x5, context, label)

        seg_out1, x_hr = self.decoder(x_hr, x[-4])
        x_hr = torch.cat([x_hr, edge_fea], dim=1)
        seg_out2 = self.cls(x_hr)

        seg_dsn = F.interpolate(seg_dsn, 
                              size=(x_.size(2), x_.size(3)), 
                              mode="bilinear", 
                              align_corners=True)
        seg_out2 = F.interpolate(seg_out2, 
                             size=(x_.size(2), x_.size(3)), 
                             mode="bilinear", 
                             align_corners=True)
        seg_out1 = F.interpolate(seg_out1, 
                             size=(x_.size(2), x_.size(3)), 
                             mode="bilinear", 
                             align_corners=True)
        edge_out = F.interpolate(edge_out, 
                                 size=(x_.size(2), x_.size(3)), 
                                 mode="bilinear", 
                                 align_corners=True)

        return seg_out1, edge_out, seg_dsn, seg_out2

