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
import sys
import cv2
import pdb
import numpy as np
import scipy.io as sio


def get_cityscapes_colors():
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    num_cls = 20
    colors = [0] * (num_cls * 3)
    colors[0:3] = (128, 64, 128)       # 0: 'road' 
    colors[3:6] = (244, 35,232)        # 1 'sidewalk'
    colors[6:9] = (70, 70, 70)         # 2''building'
    colors[9:12] = (102,102,156)       # 3 wall
    colors[12:15] =  (190,153,153)     # 4 fence
    colors[15:18] = (153,153,153)      # 5 pole
    colors[18:21] = (250,170, 30)      # 6 'traffic light'
    colors[21:24] = (220,220, 0)       # 7 'traffic sign'
    colors[24:27] = (107,142, 35)      # 8 'vegetation'
    colors[27:30] = (152,251,152)      # 9 'terrain'
    colors[30:33] = ( 70,130,180)      # 10 sky
    colors[33:36] = (220, 20, 60)      # 11 person
    colors[36:39] = (255, 0, 0)        # 12 rider
    colors[39:42] = (0, 0, 142)        # 13 car
    colors[42:45] = (0, 0, 70)         # 14 truck
    colors[45:48] = (0, 60,100)        # 15 bus
    colors[48:51] = (0, 80,100)        # 16 train
    colors[51:54] = (0, 0,230)         # 17 'motorcycle'
    colors[54:57] = (119, 11, 32)      # 18 'bicycle'
    colors[57:60] = (105, 105, 105)
    return colors


def get_ade_colors():
    colors = sio.loadmat(os.path.dirname(os.path.abspath(__file__))+'/color150.mat')['colors']
    colors = colors[:,::-1,]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0,[0,0,0])
    colors = sum(colors, [])
    return colors


def  get_pascal_context_colors():
    colors = sio.loadmat(os.path.dirname(os.path.abspath(__file__))+'/color60.mat')['color60']
    colors = colors[:,::-1,]
    colors = np.array(colors).astype(int).tolist()
    colors = sum(colors, [])
    return colors    


def get_lip_colors():
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = 20
    colors = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        colors[j * 3 + 0] = 0
        colors[j * 3 + 1] = 0
        colors[j * 3 + 2] = 0
        i = 0
        while lab:
            colors[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            colors[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            colors[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return colors


def get_cocostuff_colors():
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = 171
    colors = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        colors[j * 3 + 0] = 0
        colors[j * 3 + 1] = 0
        colors[j * 3 + 2] = 0
        i = 0
        while lab:
            colors[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            colors[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            colors[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return colors


def get_pascal_voc_colors():
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )