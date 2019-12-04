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
import matplotlib
matplotlib.use('Agg')

import torch
import os
import sys
import pdb
import cv2
import numpy as np
from torch import nn
from torch.nn import functional as F
import functools

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image as PILImage


torch_ver = torch.__version__[:3]

ignore_label = 255
id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

class_name_dict = {0:'road', 1:'sidewalk', 2:'building', 3:'wall', 4:'fence', 5:'pole',
                   6:'trafficlight', 7:'trafficsign', 8:'vegetation', 9:'terrian', 10:'sky', 
                   11:'person', 12:'rider', 13:'car', 14:'truck', 15:'bus', 16:'train',
                   17:'motorcycle', 18:'bicycle', 255: 'none'}


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    palette = [0] * (num_cls * 3)
    palette[0:3] = (128, 64, 128)       # 0: 'road' 
    palette[3:6] = (244, 35,232)        # 1 'sidewalk'
    palette[6:9] = (70, 70, 70)         # 2''building'
    palette[9:12] = (102,102,156)       # 3 wall
    palette[12:15] =  (190,153,153)     # 4 fence
    palette[15:18] = (153,153,153)      # 5 pole
    palette[18:21] = (250,170, 30)      # 6 'traffic light'
    palette[21:24] = (220,220, 0)       # 7 'traffic sign'
    palette[24:27] = (107,142, 35)      # 8 'vegetation'
    palette[27:30] = (152,251,152)      # 9 'terrain'
    palette[30:33] = ( 70,130,180)      # 10 sky
    palette[33:36] = (220, 20, 60)      # 11 person
    palette[36:39] = (255, 0, 0)        # 12 rider
    palette[39:42] = (0, 0, 142)        # 13 car
    palette[42:45] = (0, 0, 70)         # 14 truck
    palette[45:48] = (0, 60,100)        # 15 bus
    palette[48:51] = (0, 80,100)        # 16 train
    palette[51:54] = (0, 0,230)         # 17 'motorcycle'
    palette[54:57] = (119, 11, 32)      # 18 'bicycle'
    palette[57:60] = (105, 105, 105)
    return palette

palette = get_palette(20)

def id2trainId(label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

def down_sample_target(target, scale):
    row, col = target.shape
    step = scale
    r_target = target[0:row:step, :]  
    c_target = r_target[:, 0:col:step]
    return c_target


def visualize_map(atten, shape, out_path):
    atten_np = atten.cpu().data.numpy() # c x hw
    (h, w) = shape
    for row in range(2):
        for col in range(9):
            # plt.subplot(5,8,9+row*8+col)
            # pdb.set_trace()
            cm = atten_np[row*8+col] 
            cm = np.reshape(cm, (h, w))
            plt.tight_layout()
            plt.imshow(cm, cmap='Blues', interpolation='nearest')
            plt.axis('off')  
            plt.savefig(out_path+'regionmap_'+str(row*8+col)+'png', bbox_inches='tight', pad_inches = 0)
    pdb.set_trace()


def Vis_A2_Atten(img_path,
                   label_path,
                   image,
                   label,
                   atten, 
                   shape,
                   cmap=plt.cm.Blues,
                   index=1,
                   choice=1,
                   maps_count=32):
    """
    This function prints and plots the attention weight matrix.
    Input:
        choice: 1 represents plotting the histogram of the weights' distribution
                2 represents plotting the attention weights' map
    """
    atten_np = atten.cpu().data.numpy() # c x hw
    (h, w) = shape

    if choice == 1:
        # read image/ label from the given paths
        image = cv2.imread(img_path[index], cv2.IMREAD_COLOR) #1024x2048x3
        image = image[:, :, -1]
        image = cv2.resize(image, dsize=(h, w),interpolation=cv2.INTER_CUBIC)
        label = cv2.imread(label_path[index], cv2.IMREAD_GRAYSCALE) #1024x2048
        label = id2trainId(label, id_to_trainid)
        label = down_sample_target(label, 8)

    else:
        # use the image crop directly.
        image = image.astype(np.float)[index] #3x1024x2048
        image = np.transpose(image, (1,2,0))
        mean = (102.9801, 115.9465, 122.7717)
        image += mean
        image = image.astype(np.uint8)
        image = cv2.resize(image, dsize=(w, h),interpolation=cv2.INTER_CUBIC)
        label = label.cpu().numpy().astype(np.uint8)[index]
        label = down_sample_target(label, 8)

    img_label = PILImage.fromarray(label)
    img_label.putpalette(palette)

    plt.tight_layout()
    plt.figure(figsize=(48, 24))
    plt.axis('off')

    plt.subplot(5,8,1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(5,8,2)
    plt.imshow(img_label)
    plt.axis('off')

    for row in range(4):
        for col in range(8):
            plt.subplot(5,8,9+row*8+col)
            cm = atten_np[row*8+col]
            cm = np.reshape(cm, (h, w))
            plt.imshow(cm, cmap='Blues', interpolation='nearest')
            plt.axis('off')
            plt.gca().set_title("Attention Map %d" %(row*8+col))
  
    # plt.subplot(3,7,1)
    # plt.imshow(image)
    # plt.axis('off')

    # plt.subplot(3,7,2)
    # plt.imshow(img_label)
    # plt.axis('off')

    # for row in range(3):
    #     for col in range(7):
    #         if (row*7+col) == 0 or (row*7+col) == 1:
    #             continue
    #         plt.subplot(3,7,row*7+col+1)
    #         cm = atten_np[row*7+col-2]
    #         cm = np.reshape(cm, (h, w))
    #         plt.imshow(cm, cmap='Blues', interpolation='nearest')
    #         plt.axis('off')
    #         plt.gca().set_title("Attention Map %d" %(row*7+col-2))

    plt.show()
    outpath='./object_context_vis/a2map_32/'
    plt.savefig(outpath+'a2map_'+str(img_path[0][0:-3].split('/')[-1])+'png', bbox_inches='tight', pad_inches = 0)
    print("image id: {}".format(img_path[0][0:-3].split('/')[-1]))


def Vis_FastOC_Atten(img_path,
                   label_path,
                   image,
                   label,
                   atten, 
                   shape,
                   cmap=plt.cm.Blues,
                   index=1,
                   choice=1,
                   subplot=False):
    """
    This function prints and plots the attention weight matrix.
    Input:
        choice: 1 represents plotting the histogram of the weights' distribution
                2 represents plotting the attention weights' map
    """
    atten_np = atten.cpu().data.numpy() # c x hw
    (h, w) = shape

    if choice == 1:
        # read image/ label from the given paths
        image = cv2.imread(img_path[index], cv2.IMREAD_COLOR) #1024x2048x3
        image = image[:, :, -1]
        image = cv2.resize(image, dsize=(h, w),interpolation=cv2.INTER_CUBIC)
        label = cv2.imread(label_path[index], cv2.IMREAD_GRAYSCALE) #1024x2048
        label = id2trainId(label, id_to_trainid)
        label = down_sample_target(label, 8)

    else:
        # use the image crop directly.
        image = image.astype(np.float)[index] #3x1024x2048
        image = np.transpose(image, (1,2,0))
        mean = (102.9801, 115.9465, 122.7717)
        image += mean
        image = image.astype(np.uint8)
        image = cv2.resize(image, dsize=(w, h),interpolation=cv2.INTER_CUBIC)
        label = label.cpu().numpy().astype(np.uint8)[index]
        label = down_sample_target(label, 8)

    img_label = PILImage.fromarray(label)
    img_label.putpalette(palette)

    plt.tight_layout()
    plt.figure(figsize=(48, 24))
    plt.axis('off')
 
    if subplot: 
        plt.subplot(3,7,1)
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(3,7,2)
        plt.imshow(img_label)
        plt.axis('off')

    for row in range(3):
        for col in range(7):
            if (row*7+col) == 0 or (row*7+col) == 1:
                continue
            if subplot:
                plt.subplot(3,7,row*7+col+1)
            cm = atten_np[row*7+col-2]
            cm = np.reshape(cm, (h, w))
            plt.imshow(cm, cmap='Blues', interpolation='nearest')
            plt.axis('off')
            if not subplot:
                plt.show()
                outpath='./object_context_vis/fast_baseoc_map/'
                plt.savefig(outpath+'fast_baseoc_map_'+str(img_path[0][0:-3].split('/')[-1])+'_'+str(row*7+col-2)+'.png', bbox_inches='tight', pad_inches = 0)
            else:
                plt.gca().set_title("Attention Map %d" %(row*7+col-2))

    if subplot:
        plt.show()
        outpath='./object_context_vis/fast_baseoc_map/'
        plt.savefig(outpath+'fast_baseoc_map_'+str(img_path[0][0:-3].split('/')[-1])+'png', bbox_inches='tight', pad_inches = 0)
    print("image id: {}".format(img_path[0][0:-3].split('/')[-1]))

