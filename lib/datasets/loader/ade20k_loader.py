##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: DonnyYou
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import torch
from torch.utils import data

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log


class ADE20KLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_list, self.label_list, self.size_list = self.__list_dirs(root_dir, dataset)

    def __len__(self):
        return len(self.img_list)

    def _get_batch_per_gpu(self, cur_index):
        img = ImageHelper.read_image(self.img_list[cur_index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        labelmap = ImageHelper.read_image(self.label_list[cur_index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
        img_size = self.size_list[cur_index]
        img_out = [img]
        label_out = [labelmap]
        for i in range(self.configer.get('train', 'batch_per_gpu')-1):
            while True:
                cur_index = (cur_index + random.randint(1, len(self.img_list) - 1)) % len(self.img_list)
                now_img_size = self.size_list[cur_index]
                now_mark = 0 if now_img_size[0] > now_img_size[1] else 1
                mark = 0 if img_size[0] > img_size[1] else 1
                if now_mark == mark:
                    img = ImageHelper.read_image(self.img_list[cur_index],
                                                 tool=self.configer.get('data', 'image_tool'),
                                                 mode=self.configer.get('data', 'input_mode'))
                    img_out.append(img)
                    labelmap = ImageHelper.read_image(self.label_list[cur_index],
                                                      tool=self.configer.get('data', 'image_tool'), mode='P')
                    label_out.append(labelmap)
                    break

        return img_out, label_out

    def __getitem__(self, index):
        img_out, label_out = self._get_batch_per_gpu(index)
        img_list = []
        labelmap_list = []
        for img, labelmap in zip(img_out, label_out):
            if self.configer.exists('data', 'label_list'):
                labelmap = self._encode_label(labelmap)

            if self.configer.exists('data', 'reduce_zero_label'):
                labelmap = self._reduce_zero_label(labelmap)

            # process for the pascal-voc dataset
            # ori_target = ImageHelper.tonp(labelmap)
            # ori_target[ori_target == 255] = -1

            if self.aug_transform is not None:
                img, labelmap = self.aug_transform(img, labelmap=labelmap)

            if self.img_transform is not None:
                img = self.img_transform(img)

            if self.label_transform is not None:
                labelmap = self.label_transform(labelmap)

            img_list.append(img)
            labelmap_list.append(labelmap)

        border_width = [sample.size(2) for sample in img_list]
        border_height = [sample.size(1) for sample in img_list]
        target_width, target_height = max(border_width), max(border_height)
        if 'fit_stride' in self.configer.get('train', 'data_transformer'):
            stride = self.configer.get('train', 'data_transformer')['fit_stride']
            pad_w = 0 if (target_width % stride == 0) else stride - (target_width % stride)  # right
            pad_h = 0 if (target_height % stride == 0) else stride - (target_height % stride)  # down
            target_width = target_width + pad_w
            target_height = target_height + pad_h

        batch_images = torch.zeros(self.configer.get('train', 'batch_per_gpu'), 3, target_height, target_width)
        batch_labels = torch.ones(self.configer.get('train', 'batch_per_gpu'), target_height, target_width)
        batch_labels = (batch_labels * -1).long()
        for i, (img, labelmap) in enumerate(zip(img_list, labelmap_list)):
            pad_width = target_width - img.size(2)
            pad_height = target_height - img.size(1)
            if self.configer.get('train', 'data_transformer')['pad_mode'] == 'random':
                left_pad = random.randint(0, pad_width)  # pad_left
                up_pad = random.randint(0, pad_height)  # pad_up
            else:
                left_pad = 0
                up_pad = 0

            batch_images[i, :, up_pad:up_pad+img.size(1), left_pad:left_pad+img.size(2)] = img
            batch_labels[i, up_pad:up_pad+labelmap.size(0), left_pad:left_pad+labelmap.size(1)] = labelmap

        return dict(
            img=DataContainer(batch_images, stack=False),
            labelmap=DataContainer(batch_labels, stack=False),
        )

    def _reduce_zero_label(self, labelmap):
        if not self.configer.get('data', 'reduce_zero_label'):
            return labelmap

        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        label_list = list()
        size_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for file_name in os.listdir(label_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
            label_path = os.path.join(label_dir, file_name)
            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} not exists.'.format(label_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            img = ImageHelper.read_image(img_path,
                                         tool=self.configer.get('data', 'image_tool'),
                                         mode=self.configer.get('data', 'input_mode'))
            size_list.append(ImageHelper.get_size(img))

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            image_dir = os.path.join(root_dir, 'val/image')
            label_dir = os.path.join(root_dir, 'val/label')
            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                img = ImageHelper.read_image(img_path,
                                             tool=self.configer.get('data', 'image_tool'),
                                             mode=self.configer.get('data', 'input_mode'))
                size_list.append(ImageHelper.get_size(img))

        return img_list, label_list, size_list


if __name__ == "__main__":
    # Test cityscapes loader.
    pass
