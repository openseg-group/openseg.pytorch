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
from lib.models.nets.deeplabv3 import DeepLabV3

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Our approaches including FCN baseline, HRNet, OCNet, ISA, OCR
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FCN baseline 
from lib.models.nets.fcnet import FcnNet

# OCR
from lib.models.nets.ocrnet import SpatialOCRNet, ASPOCRNet
from lib.models.nets.ideal_ocrnet import IdealSpatialOCRNet, IdealSpatialOCRNetB, IdealSpatialOCRNetC, IdealGatherOCRNet, IdealDistributeOCRNet

# HRNet
from lib.models.nets.hrnet import HRNet_W48, HRNet_W48_PSP, HRNet_W48_ASPP
from lib.models.nets.hrnet import HRNet_W48_OCR, HRNet_W48_ISA, HRNet_W48_ASPOCR, HRNet_W48_OCR_B, HRNet_W48_OCR_C, HRNet_W48_OCR_D

# OCNet
from lib.models.nets.ocnet import BaseOCNet, AspOCNet

# CE2P
from lib.models.nets.ce2pnet import CE2P_OCRNet, CE2P_IdealOCRNet, CE2P_ASPOCR

from lib.utils.tools.logger import Logger as Log

SEG_MODEL_DICT = {
    # OCNet series
    'base_ocnet': BaseOCNet,
    'asp_ocnet': AspOCNet,
    # OCR series
    'spatial_ocrnet': SpatialOCRNet,
    'spatial_asp_ocrnet': ASPOCRNet,
    # OCR series with ground-truth   
    'ideal_spatial_ocrnet': IdealSpatialOCRNet,
    'ideal_spatial_ocrnet_b': IdealSpatialOCRNetB,
    'ideal_spatial_ocrnet_c': IdealSpatialOCRNetC, 
    'ideal_gather_ocrnet': IdealGatherOCRNet,
    'ideal_distribute_ocrnet': IdealDistributeOCRNet,
    # HRNet series
    'hrnet_w48': HRNet_W48,
    'hrnet_w48_psp': HRNet_W48_PSP, 
    'hrnet_w48_aspp': HRNet_W48_ASPP, 
    'hrnet_w48_isa': HRNet_W48_ISA,
    'hrnet_w48_ocr': HRNet_W48_OCR,
    'hrnet_w48_ocr_b': HRNet_W48_OCR_B,
    'hrnet_w48_ocr_c': HRNet_W48_OCR_C,
    'hrnet_w48_ocr_d': HRNet_W48_OCR_D,
    'hrnet_w48_asp_ocr': HRNet_W48_ASPOCR,
    # CE2P series
    'ce2p_asp_ocrnet': CE2P_ASPOCR,
    'ce2p_ocrnet': CE2P_OCRNet,
    'ce2p_ideal_ocrnet': CE2P_IdealOCRNet, 
    # baseline series
    'fcnet': FcnNet,
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
