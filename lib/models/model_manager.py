##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
## Copyright (c) 2019
## yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Reproduce the approaches of other papers
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from lib.models.nets.pspnet import PSPNet
from lib.models.nets.deeplabv3 import DeepLabV3, DeepLabV3_CRF


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Our approaches including FCN baseline, HRNet, OCNet, ISA, OCR, Offset Approch
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FCN baseline 
from lib.models.nets.fcnet import FcnNet

# OCR
from lib.models.nets.ocrnet import SpatialOCRNet, SpatialOCRNet_GTOffset, ChannelOCRNet, RegionSpatialOCRNet
from lib.models.nets.ocrnet import SpatialOCRNetB
from lib.models.nets.ocrnet import ASPOCRNet

# HRNet
from lib.models.nets.hrnet import HRNet_W48, HRNet_W48_B, HRNet_W48_PSP, HRNet_W48_ASPP
from lib.models.nets.hrnet import HRNet_W48_OCR, HRNet_W48_ISA, HRNet_W48_ASPOCR, HRNet_W48_OCR_B, HRNet_W48_OCR_C, HRNet_W48_OCR_D

# OCNet
from lib.models.nets.ocnet import BaseOCNet, BaseNOCNet, PyramidOCNet, AspOCNet

# ISA
from lib.models.nets.isanet import ISANet, Pyramid_ISANet, Asp_ISANet

# Fast-OCNet
from lib.models.nets.fast_ocnet import FastBaseOCNet, FastBaseNOCNet

from lib.utils.tools.logger import Logger as Log

SEG_MODEL_DICT = {
    # OCNet series
    'base_ocnet': BaseOCNet,
    'base_nocnet': BaseNOCNet,
    'pyramid_ocnet': PyramidOCNet,
    'asp_ocnet': AspOCNet,
    # ISA series
    'isanet': ISANet,
    'pyramid_isanet': Pyramid_ISANet,
    'asp_isanet': Asp_ISANet,
    'channel_ocrnet': ChannelOCRNet,
    'spatial_ocrnet': SpatialOCRNet,
    'spatial_asp_ocrnet': ASPOCRNet,
    # HRNet series
    'hrnet_w48': HRNet_W48,
    'hrnet_w48_b': HRNet_W48_B,
    'hrnet_w48_psp': HRNet_W48_PSP, 
    'hrnet_w48_aspp': HRNet_W48_ASPP, 
    'hrnet_w48_isa': HRNet_W48_ISA,
    'hrnet_w48_ocr': HRNet_W48_OCR,
    'hrnet_w48_ocr_b': HRNet_W48_OCR_B,
    'hrnet_w48_ocr_c': HRNet_W48_OCR_C,
    'hrnet_w48_ocr_d': HRNet_W48_OCR_D,
    'hrnet_w48_asp_ocr': HRNet_W48_ASPOCR,
    # baseline series
    'fcnet': FcnNet,
    'pspnet': PSPNet,
    'deeplabv3': DeepLabV3,
}


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model
