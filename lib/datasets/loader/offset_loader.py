##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: JingyiXie, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import torch 
import numpy as np
import scipy.io as io
from torch.utils import data

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log
from lib.utils.helpers.offset_helper import DTOffsetHelper

class DTOffsetLoader(data.Dataset):
    """
    Load [image, label, offset, boundary, name]
    """
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_list, self.label_list, self.offset_list, self.name_list = self.__list_dirs(root_dir, dataset)
        self.root_dir = root_dir
        self.dataset = dataset
        # check whether or not stack the data
        size_mode = self.configer.get(self.dataset, 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'

    def __len__(self):
        return len(self.img_list)

    def _load_maps(self, filename, labelmap):
        dct = self._load_mat(filename)
        distance_map = dct['depth'].astype(np.int32)
        dir_deg = dct['dir_deg'].astype(np.float)  # in [0, 360 / deg_reduce]
        deg_reduce = dct['deg_reduce'][0][0]
       
        dir_deg = deg_reduce * dir_deg - 180  # in [-180, 180]

        return distance_map, dir_deg

    def load_boundary(self, fn):
        if fn.endswith('mat'):
            mat = io.loadmat(fn)
            if 'depth' in mat:
                dist_map, _ = self._load_maps(fn, None)
                boundary_map = DTOffsetHelper.distance_to_mask_label(dist_map, np.zeros_like(dist_map)).astype(np.float32)
            else:
                boundary_map = mat['mat'].transpose(1, 2, 0)
        else:
            boundary_map = ImageHelper.read_image(fn,
                                        tool=self.configer.get('data', 'image_tool'), mode='P')
            boundary_map = boundary_map.astype(np.float32) / 255

        return boundary_map

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        labelmap = ImageHelper.read_image(self.label_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
        if self.configer.exists('data', 'label_list'):
            labelmap = self._encode_label(labelmap)
        distance_map, angle_map = self._load_maps(self.offset_list[index], labelmap)

        if self.configer.exists('data', 'reduce_zero_label') and self.configer.get('data', 'reduce_zero_label') == True:
            labelmap = self._reduce_zero_label(labelmap)

        ori_target = ImageHelper.tonp(labelmap).astype(np.int)
        ori_target[ori_target == 255] = -1
        ori_distance_map = np.array(distance_map)
        ori_angle_map = np.array(angle_map)

        if self.aug_transform is not None:
            img, labelmap, distance_map, angle_map = self.aug_transform(img, labelmap=labelmap, distance_map=distance_map, angle_map=angle_map)

        old_img = img
        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)
            distance_map = torch.from_numpy(distance_map)
            angle_map = torch.from_numpy(angle_map)

        if set(self.configer.get('val_trans', 'trans_seq')) & set(['random_crop', 'crop']):
            ori_target = labelmap.numpy()
            ori_distance_map = distance_map.numpy()
            ori_angle_map = angle_map.numpy()
            img_size = ori_target.shape[:2][::-1]
            
        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target,
            ori_distance_map=ori_distance_map,
            ori_angle_map=ori_angle_map,
            basename=os.path.basename(self.label_list[index])
        )

        return dict(
            img=DataContainer(img, stack=self.is_stack),
            labelmap=DataContainer(labelmap, stack=self.is_stack),
            distance_map=DataContainer(distance_map, stack=self.is_stack),
            angle_map=DataContainer(angle_map, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(self.name_list[index], stack=False, cpu_only=True),
        )

    def _load_mat(self, filename):
        return io.loadmat(filename)

    def _replace_ext(self, filename, ext):
        return '.'.join([filename.rpartition('.')[0], ext])

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

        if os.environ.get('use_cityscapes_style'):
            if 'GTA5_small' in root_dir:
                root_dir = root_dir.replace('GTA5_small', 'GTA5_Cityscapes')
            else:
                root_dir = root_dir.replace('GTA5', 'GTA5_Cityscapes')
            Log.info_once('Using Cityscapes style, switch to {}'.format(root_dir))
        else:
            Log.info_once('Using default root dir: {}'.format(root_dir))

        img_list = list()
        label_list = list()
        offset_list = list()
        name_list = list()

        image_subdir = os.environ.get('image_subdir', 'image')
        label_subdir = os.environ.get('label_dir', 'label')
        Log.info_once('Using label dir: {}'.format(label_subdir))
        offset_subdir = os.environ.get('offset_dir', 'dt_offset')
        Log.info_once('Using distance transform based offset: {}'.format(offset_subdir))

        image_dir = os.path.join(root_dir, dataset, image_subdir)
        label_dir = os.path.join(root_dir, dataset, label_subdir)
        offset_dir = os.path.join(root_dir, dataset, offset_subdir)

        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        file_list_txt = os.environ.get('use_file_list')
        if file_list_txt is None:
            Log.info_once('Using file list: all')
            files = sorted(os.listdir(label_dir))
        else:
            Log.info_once('Using file list: {}'.format(file_list_txt))
            with open(os.path.join(root_dir, dataset, 'file_list', file_list_txt)) as f:
                files = [x.strip() for x in f]
                
        if os.environ.get('chunk'):
            n, i = map(int, os.environ.get('chunk').split('_'))
            step = len(files) // n + 4
            files = files[step * i: step * (i + 1)]

        for file_name in files:
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
            label_path = os.path.join(label_dir, file_name)
            offset_path = os.path.join(offset_dir, self._replace_ext(file_name, 'mat'))

            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} not exists.'.format(label_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            offset_list.append(offset_path)
            name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            Log.info_once('Include val set for training ...')

            image_dir = os.path.join(root_dir, 'val', image_subdir)
            label_dir = os.path.join(root_dir, 'val', label_subdir)
            offset_dir = os.path.join(root_dir, 'val', offset_subdir)

            if file_list_txt is None:
                files = sorted(os.listdir(label_dir))
            else:
                with open(os.path.join(root_dir, 'val', 'file_list', file_list_txt)) as f:
                    files = [x.strip() for x in f]

            for file_name in files:
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                offset_path = os.path.join(offset_dir, self._replace_ext(file_name, 'mat'))
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                offset_list.append(offset_path)
                name_list.append(image_name)

        return img_list, label_list, offset_list, name_list


class SWOffsetLoader(data.Dataset):
    def __init__(self, root_dir, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.img_list, self.label_list, self.offset_h_list, self.offset_w_list, self.name_list = self.__list_dirs(root_dir, dataset)
        self.root_dir = root_dir
        self.dataset = dataset
        # check whether or not stack the data
        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        img_size = ImageHelper.get_size(img)
        labelmap = ImageHelper.read_image(self.label_list[index],
                                          tool=self.configer.get('data', 'image_tool'), mode='P')
        offsetmap_h = self._load_mat(self.offset_h_list[index])
        offsetmap_w = self._load_mat(self.offset_w_list[index])

        if os.environ.get('train_no_offset') and self.dataset == 'train':
            offsetmap_h = np.zeros_like(offsetmap_h)
            offsetmap_w = np.zeros_like(offsetmap_w)

        if self.configer.exists('data', 'label_list'):
            labelmap = self._encode_label(labelmap)

        if self.configer.exists('data', 'reduce_zero_label') and self.configer.get('data', 'reduce_zero_label') == True:
            labelmap = self._reduce_zero_label(labelmap)

        # Log.info('use dataset {}'.format(self.configer.get('dataset')))
        ori_target = ImageHelper.tonp(labelmap).astype(np.int)
        ori_target[ori_target == 255] = -1
        ori_offset_h = np.array(offsetmap_h)
        ori_offset_w = np.array(offsetmap_w)

        if self.aug_transform is not None:
            img, labelmap, offsetmap_h, offsetmap_w = self.aug_transform(img, labelmap=labelmap, offset_h_map=offsetmap_h, offset_w_map=offsetmap_w)

        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)
            offsetmap_h = torch.from_numpy(np.array(offsetmap_h)).long()
            offsetmap_w = torch.from_numpy(np.array(offsetmap_w)).long()

        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target,
            ori_offset_h=ori_offset_h,
            ori_offset_w=ori_offset_w,
        )

        return dict(
            img=DataContainer(img, stack=self.is_stack),
            labelmap=DataContainer(labelmap, stack=self.is_stack),
            offsetmap_h=DataContainer(offsetmap_h, stack=self.is_stack),
            offsetmap_w=DataContainer(offsetmap_w, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(self.name_list[index], stack=False, cpu_only=True),
        )

    def _load_mat(self, filename):
        return io.loadmat(filename)['mat']

    def _replace_ext(self, filename, ext):
        return '.'.join([filename.rpartition('.')[0], ext])

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
        offset_h_list = list()
        offset_w_list = list()
        name_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')
        label_dir = os.path.join(root_dir, dataset, 'label')
        offset_h_dir = None
        offset_w_dir = None

        subdir = os.environ.get('offset_dir')
        if subdir is not None:
            Log.info_once('Using offset dir: {}'.format(subdir))
            offset_h_dir = os.path.join(root_dir, dataset, subdir, 'h')
            offset_w_dir = os.path.join(root_dir, dataset, subdir, 'w')
        else:
            offset_type = self.configer.get('data', 'offset_type')
            assert(offset_type is not None)
            offset_h_dir = os.path.join(root_dir, dataset, offset_type, 'h')
            offset_w_dir = os.path.join(root_dir, dataset, offset_type, 'w')

        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for file_name in os.listdir(label_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
            label_path = os.path.join(label_dir, file_name)
            offset_h_path = os.path.join(offset_h_dir, self._replace_ext(file_name, 'mat'))
            offset_w_path = os.path.join(offset_w_dir, self._replace_ext(file_name, 'mat'))

            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} not exists.'.format(label_path))
                continue

            img_list.append(img_path)
            label_list.append(label_path)
            offset_h_list.append(offset_h_path)
            offset_w_list.append(offset_w_path)
            name_list.append(image_name)

        if dataset == 'train' and self.configer.get('data', 'include_val'):
            image_dir = os.path.join(root_dir, 'val/image')
            label_dir = os.path.join(root_dir, 'val/label')

            subdir = os.environ.get('offset_dir')
            if subdir is not None:
                Log.info_once('Using offset dir: {}'.format(subdir))
                offset_h_dir = os.path.join(root_dir, 'val', subdir, 'h')
                offset_w_dir = os.path.join(root_dir, 'val', subdir, 'w')
            else:
                offset_type = self.configer.get('data', 'offset_type')
                assert(offset_type is not None)
                offset_h_dir = os.path.join(root_dir, 'val', offset_type, 'h')
                offset_w_dir = os.path.join(root_dir, 'val', offset_type, 'w')

            for file_name in os.listdir(label_dir):
                image_name = '.'.join(file_name.split('.')[:-1])
                img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
                label_path = os.path.join(label_dir, file_name)
                offset_h_path = os.path.join(offset_h_dir, self._replace_ext(file_name, 'mat'))
                offset_w_path = os.path.join(offset_w_dir, self._replace_ext(file_name, 'mat'))
                if not os.path.exists(label_path) or not os.path.exists(img_path):
                    Log.error('Label Path: {} not exists.'.format(label_path))
                    continue

                img_list.append(img_path)
                label_list.append(label_path)
                offset_h_list.append(offset_h_path)
                offset_w_list.append(offset_w_path)
                name_list.append(image_name)

        return img_list, label_list, offset_h_list, offset_w_list, name_list


class SWOffsetTestLoader(data.Dataset):
    def __init__(self, root_dir, dataset='val', img_transform=None, configer=None):
        self.configer = configer
        self.img_transform = img_transform
        self.img_list, self.offset_h_list, self.offset_w_list, self.name_list = self.__list_dirs(root_dir, dataset)

        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = (size_mode != 'diverse_size')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = ImageHelper.read_image(self.img_list[index],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))
        offsetmap_h = self._load_mat(self.offset_h_list[index])
        offsetmap_w = self._load_mat(self.offset_w_list[index])
        img_size = ImageHelper.get_size(img)
        if self.img_transform is not None:
            img = self.img_transform(img)
        meta = dict(
            ori_img_size=img_size,
            border_size=img_size,
        )
        return dict(
            img=DataContainer(img, stack=self.is_stack),
            offsetmap_h=DataContainer(offsetmap_h, stack=self.is_stack),
            offsetmap_w=DataContainer(offsetmap_w, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(self.name_list[index], stack=False, cpu_only=True),
        )

    def _load_mat(self, filename):
        return io.loadmat(filename)['mat']

    def _replace_ext(self, filename, ext):
        return '.'.join([filename.rpartition('.')[0], ext])

    def __list_dirs(self, root_dir, dataset):
        img_list = list()
        offset_h_list = list()
        offset_w_list = list()
        name_list = list()
        image_dir = os.path.join(root_dir, dataset, 'image')

        offset_h_dir = None
        offset_w_dir = None

        offset_type = self.configer.get('data', 'offset_type')
        assert(offset_type is not None)
        offset_h_dir = os.path.join(root_dir, dataset, offset_type, 'h')
        offset_w_dir = os.path.join(root_dir, dataset, offset_type, 'w') 
        img_extension = os.listdir(image_dir)[0].split('.')[-1]

        for file_name in os.listdir(label_dir):
            image_name = '.'.join(file_name.split('.')[:-1])
            img_path = os.path.join(image_dir, '{}.{}'.format(image_name, img_extension))
            offset_h_path = os.path.join(offset_h_dir, self._replace_ext(file_name, 'mat'))
            offset_w_path = os.path.join(offset_w_dir, self._replace_ext(file_name, 'mat'))

            if not os.path.exists(label_path) or not os.path.exists(img_path):
                Log.error('Label Path: {} not exists.'.format(label_path))
                continue
            img_list.append(img_path)
            offset_h_list.append(offset_h_path)
            offset_w_list.append(offset_w_path)
            name_list.append(image_name)

        return img_list, offset_h_list, offset_w_list, name_list


def load_mat(filename):
    return io.loadmat(filename)['mat']

def replace_ext(filename, ext):
    return '.'.join([filename.rpartition('.')[0], ext])


if __name__ == "__main__":
    pass