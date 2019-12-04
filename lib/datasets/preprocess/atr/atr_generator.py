#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: LayneH
# COCO det data generator.

'''


'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import argparse
import shutil
import numpy as np
import PIL.Image as Image
import cv2


LABEL_DIR = 'label'
IMAGE_DIR = 'image'
EDGE_DIR = 'edge'


def gen_edge(label):
    sobelx = cv2.Sobel(label, cv2.CV_64F, 1, 0)          # Find x and y gradients
    sobely = cv2.Sobel(label, cv2.CV_64F, 0, 1)

    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    # angle = np.arctan2(sobely, sobelx) * (180 / np.pi)

    magnitude[np.where(magnitude > 0)] = 1
    return magnitude.astype(np.uint8)


class ATRGenerator(object):
    def __init__(self, args, image_dir=IMAGE_DIR, label_dir=LABEL_DIR, edge_dir=EDGE_DIR):
        self.args = args
        self.train_label_dir = os.path.join(self.args.save_dir, 'train', label_dir)
        if not os.path.exists(self.train_label_dir):
            os.makedirs(self.train_label_dir)

        self.train_edge_dir = os.path.join(self.args.save_dir, 'train', edge_dir)
        if not os.path.exists(self.train_edge_dir):
            os.makedirs(self.train_edge_dir)

    def generate_label(self):
        trans_idx = np.array([0, 1, 2, 4, 5, 12, 9, 6, 255, 18, 19, 13, 16, 17, 14, 15, 255, 11], dtype=np.uint8)

        train_mask_folder = os.path.join(self.args.ori_root_dir, 'SegmentationClassAug')

        for filename in os.listdir(train_mask_folder):
            print(filename)
            if filename.endswith(".png"):
                maskpath = os.path.join(train_mask_folder, filename)
                if os.path.isfile(maskpath):
                    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
                    mask = trans_idx[mask]
                    edge = gen_edge(mask)
                    cv2.imwrite(os.path.join(self.train_label_dir, filename), mask.astype(np.uint8))
                    cv2.imwrite(os.path.join(self.train_edge_dir, filename), edge.astype(np.uint8))
                else:
                    print('cannot find the mask:', maskpath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_root_dir', default=None, type=str,
                        dest='ori_root_dir', help='The directory of the cityscapes data.')

    args = parser.parse_args()

    ATR_generator = ATRGenerator(args)
    ATR_generator.generate_label()