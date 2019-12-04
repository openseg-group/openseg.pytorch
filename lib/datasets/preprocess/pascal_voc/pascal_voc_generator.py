#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Lang Huang(layenhuang@outlook.com)
# Pascal Context aug data generator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import argparse
import shutil
import scipy.io as sio
import cv2
import numpy as np
import torch


LABEL_DIR = 'label'
IMAGE_DIR = 'image'


class PascalVOCGenerator(object):
    def __init__(self, args, image_dir=IMAGE_DIR, label_dir=LABEL_DIR):
        self.args = args
        self.train_label_dir = os.path.join(self.args.save_dir, 'train', label_dir)
        self.val_label_dir = os.path.join(self.args.save_dir, 'val', label_dir)
        if not os.path.exists(self.train_label_dir):
            os.makedirs(self.train_label_dir)

        if not os.path.exists(self.val_label_dir):
            os.makedirs(self.val_label_dir)

        self.train_image_dir = os.path.join(self.args.save_dir, 'train', image_dir)
        self.val_image_dir = os.path.join(self.args.save_dir, 'val', image_dir)
        if not os.path.exists(self.train_image_dir):
            os.makedirs(self.train_image_dir)

        if not os.path.exists(self.val_image_dir):
            os.makedirs(self.val_image_dir)
        
        self.train_mask = torch.load(os.path.join(self.args.ori_root_dir, "PytorchEncoding/train.pth"))
        self.val_mask = torch.load(os.path.join(self.args.ori_root_dir, "PytorchEncoding/val.pth"))
    

    def generate_label(self):
        train_img_folder = os.path.join(self.args.ori_root_dir, 'JPEGImages')
        val_img_folder = os.path.join(self.args.ori_root_dir, 'JPEGImages')

        for basename, mask in self.train_mask.items():
            basename = str(basename)
            print(basename)
            basename = basename[:4] + "_" + basename[4:]
            filename = basename + ".jpg"
            imgpath = os.path.join(train_img_folder, filename)
            shutil.copy(imgpath,
                        os.path.join(self.train_image_dir, filename))
            mask = np.asarray(mask)
            cv2.imwrite(os.path.join(self.train_label_dir, basename + ".png"), mask)
        
        for basename, mask in self.val_mask.items():
            basename = str(basename)
            print(basename)
            basename = basename[:4] + "_" + basename[4:]
            filename = basename + ".jpg"
            imgpath = os.path.join(val_img_folder, filename)
            shutil.copy(imgpath,
                        os.path.join(self.val_image_dir, filename))
            mask = np.asarray(mask)
            cv2.imwrite(os.path.join(self.val_label_dir, basename + ".png"), mask)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_root_dir', default=None, type=str,
                        dest='ori_root_dir', help='The directory of the cityscapes data.')

    args = parser.parse_args()

    pcontext_generator = PContextGenerator(args)
    pcontext_generator.generate_label()