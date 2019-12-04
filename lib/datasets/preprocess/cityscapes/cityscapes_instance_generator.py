#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Updated by: Lang Huang(laynehuang@outlook.com)
# CityScape Seg data generator.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import shutil


IMAGE_DIR = 'image'
LABEL_DIR = 'label'
INSTANCE_DIR = 'instance'

def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class CityscapesInstanceGenerator(object):

    def __init__(self, args, image_dir=IMAGE_DIR, label_dir=LABEL_DIR, instance_dir=INSTANCE_DIR):
        self.args = args        
        self.train_instance_dir = os.path.join(self.args.save_dir, 'train', instance_dir)
        self.val_instance_dir = os.path.join(self.args.save_dir, 'val', instance_dir)
        self.coarse_instance_dir = os.path.join(self.args.save_dir, 'coarse', instance_dir)
        if not os.path.exists(self.train_instance_dir):
            os.makedirs(self.train_instance_dir)
        if not os.path.exists(self.val_instance_dir):
            os.makedirs(self.val_instance_dir)
        if not os.path.exists(self.coarse_instance_dir):
            os.makedirs(self.coarse_instance_dir)

    def generate_instance(self):
        if not self.args.coarse:
            ori_train_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/train')
            ori_train_label_dir = os.path.join(self.args.ori_root_dir, 'gtFine/train')
            ori_val_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/val')
            ori_val_label_dir = os.path.join(self.args.ori_root_dir, 'gtFine/val')

            for image_file in self.__list_dir(ori_train_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                print(image_name)
                instance_file = '{}_gtFine_instanceIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_train_label_dir, instance_file),
                        os.path.join(self.train_instance_dir, '{}.png'.format(shotname)))

            for image_file in self.__list_dir(ori_val_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                print(image_name)
                instance_file = '{}_gtFine_instanceIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_val_label_dir, instance_file),
                        os.path.join(self.val_instance_dir, '{}.png'.format(shotname)))

        else:
            ori_train_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/train')
            ori_train_label_dir = os.path.join(self.args.ori_root_dir, 'gtFine/train')
            ori_train_extra_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/train_extra')
            ori_train_extra_label_dir = os.path.join(self.args.ori_root_dir, 'gtCoarse/train_extra')
            ori_val_img_dir = os.path.join(self.args.ori_root_dir, 'leftImg8bit/val')
            ori_val_label_dir = os.path.join(self.args.ori_root_dir, 'gtFine/val')

            for image_file in self.__list_dir(ori_train_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                print(image_name)
                instance_file = '{}_gtFine_instanceIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_train_label_dir, instance_file),
                        os.path.join(self.train_instance_dir, '{}.png'.format(shotname)))

            for image_file in self.__list_dir(ori_train_extra_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                print(image_name)
                instance_file = '{}_gtCoarse_instanceIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_train_extra_label_dir, instance_file),
                        os.path.join(self.coarse_instance_dir, '{}.png'.format(shotname)))

            for image_file in self.__list_dir(ori_val_img_dir):
                image_name = '_'.join(image_file.split('_')[:-1])
                print(image_name)
                instance_file = '{}_gtFine_instanceIds.png'.format(image_name)
                shotname, extension = os.path.splitext(image_file.split('/')[-1])
                shutil.copy(os.path.join(ori_val_label_dir, instance_file),
                        os.path.join(self.val_instance_dir, '{}.png'.format(shotname)))

    def __list_dir(self, dir_name):
        filename_list = list()
        for item in os.listdir(dir_name):
            if os.path.isdir(os.path.join(dir_name, item)):
                for filename in os.listdir(os.path.join(dir_name, item)):
                    filename_list.append('{}/{}'.format(item, filename))
            else:
                filename_list.append(item)
        return filename_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse', type=str2bool, nargs='?', default=False,
                        dest='coarse', help='Whether is the coarse data.')
    parser.add_argument('--save_dir', default=None, type=str,
                        dest='save_dir', help='The directory to save the data.')
    parser.add_argument('--ori_root_dir', default=None, type=str,
                        dest='ori_root_dir', help='The directory of the cityscapes data.')

    args = parser.parse_args()

    cityscapes_generator = CityscapesInstanceGenerator(args)
    cityscapes_generator.generate_instance()

# /root/miniconda3/bin/python cityscapes_instance_generator.py --coarse True \
# --save_dir /msravcshare/dataset/cityscapes/ --ori_root_dir \
# /msravcshare/yuyua/code/segmentation/deeplab_v3/dataset/cityscapes/
