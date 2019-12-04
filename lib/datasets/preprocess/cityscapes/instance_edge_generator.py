#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Lang Huang
# Generate the edge files and convert the labels of edge pixels / non-edge pixels to void.
# Small objects will be ignored.

import os
import cv2
import pdb
import glob

import numpy as np

from PIL import Image
from shutil import copyfile


def _generate_edge(label):
    h, w = label.shape
    edge = np.zeros(label.shape, dtype=np.uint8)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 255

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 255

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 255

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 255

    return edge


# def generate_edge(label, edge_width=10):
#     area_thrs = 4900
#     edge = np.zeros_like(label, dtype=np.uint8)
#     valid_contour = []
#     for i in np.unique(label):        
#         temp = (label == i).astype(np.uint8)
#         _, contours, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
#         for contour in contours:
#             # check the area
#             area = cv2.contourArea(contour)
#             if area < area_thrs:
#                 continue
            
#             # check the minimum height/width
#             rect = cv2.minAreaRect(contour)
#             w, h = rect[1]
#             if w < edge_width * 2 or h < edge_width * 2:
#                 continue
            
#             valid_contour.append(contour)
    
#     # draw valid contours as edge
#     cv2.drawContours(edge, valid_contour, -1, 255, thickness=1)
    
#     # dilation on edge
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
#     edge = cv2.dilate(edge, kernel)
#     return edge


def _get_bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return max(0, rmin - 1), min(rmax + 1, img.shape[0] - 1), max(0, cmin - 1), min(cmax + 1, img.shape[1] - 1)

def generate_edge(label, edge_width=10, area_thrs=200):
    label_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    edge = np.zeros_like(label, dtype=np.uint8)
    for i in np.unique(label):
        # have no instance
        if i < 1000 or (i // 1000) not in label_list:
            continue
        
        # filter out small objects
        mask = (label == i).astype(np.uint8)
        if mask.sum() < area_thrs:
            continue
        
        rmin, rmax, cmin, cmax = _get_bbox(mask)
        mask_edge = _generate_edge(mask[rmin:rmax+1, cmin:cmax+1])
        edge[rmin:rmax+1, cmin:cmax+1][mask_edge > 0] = 255
    
    # dilation on edge
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge


def generate_train_val_edge(label_path, edge_path, kernel_size=10, area_thrs=200):
    for label_file in os.listdir(label_path):
        print(label_file)
        label = np.array(Image.open(label_path + label_file))
        edge = generate_edge(label, kernel_size, area_thrs=area_thrs)
        
        im_edge = Image.fromarray(edge, 'P')

        edge_file = label_file.replace('label', 'edge')
        im_edge.save(edge_path + edge_file)

        # out_edge = np.array(Image.open(edge_path + edge_file).convert('P'))

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

def _encode_label(labelmap):
    label_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    encoded_labelmap = np.ones_like(labelmap, dtype=np.uint8) * 255
    for i, class_id in enumerate(label_list):
        encoded_labelmap[labelmap == class_id] = i

    return encoded_labelmap

def label_edge2void(label_path, edge_path, dest_label_path):
    '''
    Set the pixels along the edge as void label.
    Used to train the models without supervision on the edge pixels.
    '''
    for label_file in os.listdir(label_path):
        print(label_file)
        edge_file = label_file.replace('label', 'edge')

        label = np.array(Image.open(label_path + label_file).convert('P'))
        edge = np.array(Image.open(edge_path + edge_file).convert('P'))

        label[edge == 255] = 255
        # label = _encode_label(label)
        label_update = Image.fromarray(label)
        label_update.putpalette(get_cityscapes_colors())

        label_update.save(dest_label_path + label_file)


def label_nedge2void(label_path, edge_path, dest_label_path):
    '''
    Set the pixels except the edge as void label.
    Used to evaluate the performance of various models on the edge pixels.
    '''
    for label_file in os.listdir(label_path):
        print(label_file)
        edge_file = label_file.replace('label', 'edge')

        label = np.array(Image.open(label_path + label_file).convert('P'))
        edge = np.array(Image.open(edge_path + edge_file).convert('P'))

        label[edge == 0] = 255
        label_update = Image.fromarray(label)
        
        label_update.save(dest_label_path + label_file)


def calculate_edge(edge_path):
    '''
    Set the pixels except the edge as void label.
    Used to evaluate the performance of various models on the edge pixels.
    '''
    edge_cnt = 0.0
    non_edge_cnt = 0.0

    print("ratio: {:f}".format(1/2))

    for label_file in os.listdir(label_path):
        print(label_file)
        edge_file = label_file.replace('label', 'edge')
        edge = np.array(Image.open(edge_path + edge_file).convert('P'))

        edge_cnt += np.sum(edge == 255)
        non_edge_cnt += np.sum(edge == 0)

    print("ratio: {:f}".format(edge_cnt/non_edge_cnt))


if __name__ == "__main__":
    label_path = "/home/huanglang/datasets/Cityscape/val/label/"
    instance_path = "/home/huanglang/datasets/Cityscape/val/instance/"
    edge_path = "/home/huanglang/datasets/Cityscape/val/edge_instance/"
    if not os.path.exists(edge_path):
        os.makedirs(edge_path)
    
    generate_train_val_edge(instance_path, edge_path, 10, area_thrs=3600)

    label_edge2void_path = "/home/huanglang/datasets/Cityscape/edge_inst_width10/val/label_edge_void/"
    label_nedge2void_path = "/home/huanglang/datasets/Cityscape/edge_inst_width10/val/label_non_edge_void/"
    if not os.path.exists(label_edge2void_path):
        os.makedirs(label_edge2void_path)
    if not os.path.exists(label_nedge2void_path):
        os.makedirs(label_nedge2void_path)

    label_edge2void(label_path, edge_path, label_edge2void_path)
    label_nedge2void(label_path, edge_path, label_nedge2void_path)

    calculate_edge(edge_path)
