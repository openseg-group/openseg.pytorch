##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class FastBaseOCNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        elif "mobilenetv2" in self.configer.get('network', 'backbone'):
            in_channels = [160, 320]
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block import ObjectContext_Module
        self.object_head = ObjectContext_Module(self.num_classes)
        from lib.models.modules.fast_base_oc_block import FastBaseOC_Module
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, out_channels=512, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        if "mobilenetv2" in self.configer.get('network', 'backbone'):
            self.oc_module_pre = nn.Sequential(
                nn.Conv2d(in_channels[1], 256, kernel_size=3, stride=1, padding=1),
                ModuleHelper.BNReLU(256, bn_type=self.configer.get('network', 'bn_type')),
            )
            from lib.models.modules.fast_base_oc_block import ObjectContext_Module
            self.object_head = ObjectContext_Module(self.num_classes)
            from lib.models.modules.fast_base_oc_block import FastBaseOC_Module
            self.fast_oc_head = FastBaseOC_Module(in_channels=256, key_channels=256, out_channels=256, scale=1,
                                                  dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
            self.head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            self.dsn_head = nn.Sequential(
                nn.Conv2d(in_channels[0], 128, kernel_size=3, stride=1, padding=1),
                ModuleHelper.BNReLU(128, bn_type=self.configer.get('network', 'bn_type')),
                nn.Dropout2d(0.05),
                nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )


    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        context = self.object_head(x, x_dsn)
        # x, sim_map = self.fast_oc_head(x, context)
        x = self.fast_oc_head(x, context)
        x = self.head(x)

        # batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        # sim_map = sim_map.permute(0, 2, 1)
        # sim_map = sim_map.view(batch_size, c, h, w)

        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        # sim_map = F.interpolate(sim_map, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        return  x_dsn, x


class Channel_FastBaseOCNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(Channel_FastBaseOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        elif "mobilenetv2" in self.configer.get('network', 'backbone'):
            in_channels = [160, 320]
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block import ObjectContext_Channel_Module
        self.object_head = ObjectContext_Channel_Module(self.num_classes)
        from lib.models.modules.fast_base_oc_block import FastBaseOC_Module
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, out_channels=512, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )


    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        context = self.object_head(x, x_dsn)
        # x, sim_map = self.fast_oc_head(x, context)
        x = self.fast_oc_head(x, context)
        x = self.head(x)

        # batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        # sim_map = sim_map.permute(0, 2, 1)
        # sim_map = sim_map.view(batch_size, c, h, w)

        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        # sim_map = F.interpolate(sim_map, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        return  x_dsn, x


class FastBaseNOCNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseNOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block import NonObjectContext_Module
        self.object_head = NonObjectContext_Module(self.num_classes)
        from lib.models.modules.fast_base_oc_block import FastBaseOC_Module
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, out_channels=512, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        context = self.object_head(x, x_dsn)
        x = self.fast_oc_head(x, context)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


class FastBaseOCNet_double_softmax(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseOCNet_double_softmax, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block import ObjectContext_Module_double_softmax
        self.object_head = ObjectContext_Module_double_softmax(self.num_classes, scale=10)
        from lib.models.modules.fast_base_oc_block import FastBaseOC_Module
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, out_channels=512, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        context = self.object_head(x, x_dsn)

        x, sim_map = self.fast_oc_head(x, context)
        x = self.head(x)

        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        sim_map = sim_map.permute(0, 2, 1)
        sim_map = sim_map.view(batch_size, c, h, w)

        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, sim_map


class FastBaseOCNet_dual(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseOCNet_dual, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block_dual import ObjectContext_Module
        self.object_head = ObjectContext_Module(self.num_classes)
        from lib.models.modules.fast_base_oc_block_dual import FastBaseOC_Module
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, out_channels=512, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        oc_map = self.object_head(x_dsn)
        x = self.fast_oc_head(x, oc_map)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


class FastBaseOCNet_dual_v2(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseOCNet_dual_v2, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block_dual_v2 import ObjectContext_Module
        self.object_head = ObjectContext_Module(self.num_classes)
        from lib.models.modules.fast_base_oc_block_dual_v2 import FastBaseOC_Module
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, out_channels=512, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        oc_map = self.object_head(x_dsn)
        x = self.fast_oc_head(x, oc_map)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


class FastBaseOCNet_dual_v3(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseOCNet_dual_v3, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block_dual_v3 import ObjectContext_Module
        self.object_head = ObjectContext_Module(self.num_classes)
        from lib.models.modules.fast_base_oc_block_dual_v3 import FastBaseOC_Context_Module
        self.fast_oc_head = FastBaseOC_Context_Module(in_channels=512, key_channels=256, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        oc_map = self.object_head(x_dsn)
        x = self.fast_oc_head(x, oc_map)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


class FastBaseOCNet_mask(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseOCNet_mask, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block import ObjectContext_Module_onehot
        self.object_head = ObjectContext_Module_onehot(self.num_classes)
        from lib.models.modules.fast_base_oc_block import FastBaseOC_Module
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, out_channels=512, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        context = self.object_head(x, x_dsn)
        x = self.fast_oc_head(x, context)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


class FastBaseOCNet_weight(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastBaseOCNet_weight, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.oc_module_pre = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )
        from lib.models.modules.fast_base_oc_block import Weighted_ObjectContext_Module
        self.object_head = Weighted_ObjectContext_Module(self.num_classes)
        from lib.models.modules.fast_base_oc_block import FastBaseOC_Module
        self.fast_oc_head = FastBaseOC_Module(in_channels=512, key_channels=256, out_channels=512, scale=1,
                                              dropout=0.05, bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.oc_module_pre(x[-1])
        context = self.object_head(x, x_dsn)
        x = self.fast_oc_head(x, context)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x


class FastAspOCNet(nn.Module):
    def __init__(self, configer):
        self.inplanes = 128
        super(FastAspOCNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        self.down = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            )
        from lib.models.modules.fast_asp_oc_block import Fast_ASP_OC_Module
        self.fast_oc_head = Fast_ASP_OC_Module(512, 512, num_classes=self.num_classes,
                                               bn_type=self.configer.get('network', 'bn_type'))
        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.down(x[-1])
        x = self.fast_oc_head(x, x_dsn)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return x_dsn, x
