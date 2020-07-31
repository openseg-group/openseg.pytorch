#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jingyi Xie


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random

import cv2
import numpy as np

from lib.utils.tools.logger import Logger as Log
from lib.datasets.tools.transforms import DeNormalize

class _BaseTransform(object):

    DATA_ITEMS = (
        'labelmap', 'maskmap',
        'distance_map', 'angle_map', 'multi_label_direction_map',
        'boundary_map', 'offsetmap',
        # 'offsetmap_h', 'offsetmap_w', 
        'region_indexmap'
    )

    def __call__(self, img, **kwargs):

        data_dict = collections.defaultdict(lambda: None)
        data_dict.update(kwargs)

        return img, data_dict

    def _process(self, img, data_dict, skip_condition, *args, **kwargs):
        assert isinstance(img, np.ndarray), \
            "img should be numpy array, got {}.".format(type(img))
        if not skip_condition:
            img = self._process_img(img, *args, **kwargs)

        ret_dict = collections.defaultdict(lambda: None)
        for name in self.DATA_ITEMS:
            func_name = '_process_' + name
            x = data_dict[name]

            assert isinstance(x, np.ndarray) or x is None, \
                "{} should be numpy array or None, got {}.".format(
                    name, type(x))

            if hasattr(self, func_name) and x is not None and not skip_condition:
                ret_dict[name] = getattr(self, func_name)(x, *args, **kwargs)
            else:
                ret_dict[name] = x

        return img, ret_dict


class Padding(_BaseTransform):
    """ Padding the Image to proper size.
            Args:
                stride: the stride of the network.
                pad_value: the value that pad to the image border.
                img: Image object as input.
            Returns::
                img: Image object.
    """

    def __init__(self, pad=None, pad_ratio=0.5, mean=(104, 117, 123), allow_outside_center=True):
        self.pad = pad
        self.ratio = pad_ratio
        self.mean = mean
        self.allow_outside_center = allow_outside_center

    def _pad(self, x, pad_value, height, width, target_size, offset_left, offset_up):
        expand_x = np.zeros((
            max(height, target_size[1]) + abs(offset_up),
            max(width, target_size[0]) + abs(offset_left),
            *x.shape[2:]
        ), dtype=x.dtype)
        expand_x[:, :] = pad_value
        expand_x[
            abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
            abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = x
        x = expand_x[
            max(offset_up, 0):max(offset_up, 0) + target_size[1],
            max(offset_left, 0):max(offset_left, 0) + target_size[0]
        ]
        return x

    def _process_img(self, img, *args):
        return self._pad(img, self.mean, *args)

    def _process_labelmap(self, x, *args):
        return self._pad(x, 255, *args)

    def _process_region_indexmap(self, x, *args):
        return self._pad(x, 0, *args)

    def _process_maskmap(self, x, *args):
        return self._pad(x, 1, *args)

    def _process_distance_map(self, x, *args):
        return self._pad(x, 255, *args)

    def _process_angle_map(self, x, *args):
        return self._pad(x, 0, *args)

    def _process_boundary_map(self, x, *args):
        return self._pad(x, 0, *args)

    def _process_multi_label_direction_map(self, x, *args):
        return self._pad(x, 0, *args)

    # def _process_offsetmap_h(self, x, *args):
    #     return self._pad(x, 0, *args)

    # def _process_offsetmap_w(self, x, *args):
    #     return self._pad(x, 0, *args)

    def _process_offsetmap(self, x, *args):
        return self._pad(x, 0, *args)

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        height, width, channels = img.shape
        left_pad, up_pad, right_pad, down_pad = self.pad

        target_size = [width + left_pad +
                       right_pad, height + up_pad + down_pad]
        offset_left = -left_pad
        offset_up = -up_pad

        return self._process(
            img, data_dict,
            random.random() > self.ratio,
            height, width, target_size, offset_left, offset_up
        )


class RandomHFlip(_BaseTransform):
    def __init__(self, swap_pair=None, flip_ratio=0.5):
        self.swap_pair = swap_pair
        self.ratio = flip_ratio

    def _process_img(self, img):
        return cv2.flip(img, 1)

    def _process_labelmap(self, labelmap):
        labelmap = cv2.flip(labelmap, 1)
        # to handle datasets with left/right annatations
        if self.swap_pair is not None:
            assert isinstance(self.swap_pair, (tuple, list))
            temp = labelmap.copy()
            for pair in self.swap_pair:
                assert isinstance(pair, (tuple, list)) and len(pair) == 2
                labelmap[temp == pair[0]] = pair[1]
                labelmap[temp == pair[1]] = pair[0]

        return labelmap

    def _process_region_indexmap(self, labelmap):
        return cv2.flip(labelmap, 1)

    def _process_maskmap(self, x):
        return cv2.flip(x, 1)

    def _process_distance_map(self, x):
        return cv2.flip(x, 1)

    def _process_angle_map(self, angle_map):
        ret_angle_map = angle_map.copy()
        mask = (angle_map > 0) & (angle_map < 180)
        ret_angle_map[mask] = 180 - angle_map[mask]
        mask = (angle_map < 0) & (angle_map > -180)
        ret_angle_map[mask] = - (180 + angle_map[mask])
        ret_angle_map = cv2.flip(ret_angle_map, 1)
        return ret_angle_map

    def _process_boundary_map(self, x):
        return cv2.flip(x, 1)

    def _process_multi_label_direction_map(self, multi_label_direction_map):
        perm = [4, 3, 2, 1, 0, 7, 6, 5]
        multi_label_direction_map = cv2.flip(multi_label_direction_map, 1)
        multi_label_direction_map = multi_label_direction_map[..., perm]
        return multi_label_direction_map

    # def _process_offsetmap_h(self, x):
    #     return cv2.flip(x, 1)

    # def _process_offsetmap_w(self, x):
    #     return -cv2.flip(x, 1)

    def _process_offsetmap_w(self, x):
        x = cv2.flip(x, 1)
        x[..., 1] = -x[..., 1]
        return x

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        return self._process(
            img, data_dict,
            random.random() > self.ratio
        )


class RandomSaturation(_BaseTransform):
    def __init__(self, lower=0.5, upper=1.5, saturation_ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = saturation_ratio
        assert self.upper >= self.lower, "saturation upper must be >= lower."
        assert self.lower >= 0, "saturation lower must be non-negative."

    def _process_img(self, img):
        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] *= random.uniform(self.lower, self.upper)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        return self._process(
            img, data_dict,
            random.random() > self.ratio
        )


class RandomHue(_BaseTransform):
    def __init__(self, delta=18, hue_ratio=0.5):
        assert 0 <= delta <= 360
        self.delta = delta
        self.ratio = hue_ratio

    def _process_img(self, img):
        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] += random.uniform(-self.delta, self.delta)
        img[:, :, 0][img[:, :, 0] > 360] -= 360
        img[:, :, 0][img[:, :, 0] < 0] += 360
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        return self._process(
            img, data_dict,
            random.random() > self.ratio
        )


class RandomPerm(_BaseTransform):
    def __init__(self, perm_ratio=0.5):
        self.ratio = perm_ratio
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def _process_img(self, img):
        swap = self.perms[random.randint(0, len(self.perms) - 1)]
        img = img[:, :, swap].astype(np.uint8)
        return img

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        return self._process(
            img, data_dict,
            random.random() > self.ratio
        )


class RandomContrast(_BaseTransform):
    def __init__(self, lower=0.5, upper=1.5, contrast_ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = contrast_ratio
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def _process_img(self, img):
        img = img.astype(np.float32)
        img *= random.uniform(self.lower, self.upper)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        return self._process(
            img, data_dict,
            random.random() > self.ratio
        )


class RandomBrightness(_BaseTransform):
    def __init__(self, shift_value=30, brightness_ratio=0.5):
        self.shift_value = shift_value
        self.ratio = brightness_ratio

    def _process_img(self, img):
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __call__(self, img, **kwargs):
        img, data_dict = super().__call__(img, **kwargs)

        return self._process(
            img, data_dict,
            random.random() > self.ratio
        )


class RandomResize(_BaseTransform):
    """Resize the given numpy.ndarray to random size and aspect ratio.

    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, scale_range=(0.75, 1.25), aspect_range=(0.9, 1.1), target_size=None,
                 resize_bound=None, method='random', max_side_bound=None, scale_list=None, resize_ratio=0.5):
        self.scale_range = scale_range
        self.aspect_range = aspect_range
        self.resize_bound = resize_bound
        self.max_side_bound = max_side_bound
        self.scale_list = scale_list
        self.method = method
        self.ratio = resize_ratio

        if target_size is not None:
            if isinstance(target_size, int):
                self.input_size = (target_size, target_size)
            elif isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                self.input_size = target_size
            else:
                raise TypeError(
                    'Got inappropriate size arg: {}'.format(target_size))
        else:
            self.input_size = None

    def get_scale(self, img_size):
        if self.method == 'random':
            scale_ratio = random.uniform(
                self.scale_range[0], self.scale_range[1])
            return scale_ratio

        elif self.method == 'bound':
            scale1 = self.resize_bound[0] / min(img_size)
            scale2 = self.resize_bound[1] / max(img_size)
            scale = min(scale1, scale2)
            return scale

        else:
            Log.error('Resize method {} is invalid.'.format(self.method))
            exit(1)

    def _process_img(self, img, converted_size, *args):
        return cv2.resize(img, converted_size, interpolation=cv2.INTER_CUBIC).astype(np.uint8)

    def _process_labelmap(self, x, converted_size, *args):
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    def _process_region_indexmap(self, x, converted_size, *args):
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    def _process_maskmap(self, x, converted_size, *args):
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    def _process_distance_map(self, x, converted_size, *args):
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    def _process_angle_map(self, x, converted_size, *args):
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    def _process_boundary_map(self, x, converted_size, *args):
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    def _process_multi_label_direction_map(self, x, converted_size, *args):
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    # def _process_offsetmap_h(self, x, converted_size, h_scale_ratio, w_scale_ratio):
    #     return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST) * h_scale_ratio

    # def _process_offsetmap_w(self, x, converted_size, h_scale_ratio, w_scale_ratio):
    #     return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST) * w_scale_ratio

    def _process_offsetmap(self, x, converted_size, h_scale_ratio, w_scale_ratio):
        return cv2.resize(x, converted_size, interpolation=cv2.INTER_NEAREST)

    def __call__(self, img, **kwargs):
        """
        Args:
            img     (Image):   Image to be resized.
            maskmap    (Image):   Mask to be resized.
            kpt     (list):    keypoints to be resized.
            center: (list):    center points to be resized.

        Returns:
            Image:  Randomly resize image.
            Image:  Randomly resize maskmap.
            list:   Randomly resize keypoints.
            list:   Randomly resize center points.
        """
        img, data_dict = super().__call__(img, **kwargs)

        height, width, _ = img.shape
        if self.scale_list is None:
            scale_ratio = self.get_scale([width, height])
        else:
            scale_ratio = self.scale_list[random.randint(
                0, len(self.scale_list)-1)]

        aspect_ratio = random.uniform(*self.aspect_range)
        w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
        h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
        if self.max_side_bound is not None and max(height*h_scale_ratio, width*w_scale_ratio) > self.max_side_bound:
            d_ratio = self.max_side_bound / max(height * h_scale_ratio, width * w_scale_ratio)
            w_scale_ratio *= d_ratio
            h_scale_ratio *= d_ratio

        converted_size = (int(width * w_scale_ratio),
                          int(height * h_scale_ratio))
        return self._process(
            img, data_dict,
            random.random() > self.ratio,
            converted_size, h_scale_ratio, w_scale_ratio
        )


class RandomRotate(_BaseTransform):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree, rotate_ratio=0.5, mean=(104, 117, 123)):
        assert isinstance(max_degree, int)
        self.max_degree = max_degree
        self.ratio = rotate_ratio
        self.mean = mean
        Log.warn(
            'Currently `RandomRotate` is only implemented for `img`, `labelmap` and `maskmap`.')

    def _warp(self, x, border_value, rotate_mat, new_width, new_height):
        return cv2.warpAffine(x, rotate_mat, (new_width, new_height), borderValue=border_value)

    def _process_img(self, x, *args):
        return self._warp(x, self.mean, *args).astype(np.uint8)

    def _process_labelmap(self, x, *args):
        return self._warp(x, (255, 255, 255), *args).astype(np.uint8)

    def _process_maskmap(self, x, *args):
        return self._warp(x, (1, 1, 1), *args).astype(np.uint8)

    def __call__(self, img, **kwargs):
        """
        Args:
            img    (Image):     Image to be rotated.
            maskmap   (Image):     Mask to be rotated.
            kpt    (list):      Keypoints to be rotated.
            center (list):      Center points to be rotated.

        Returns:
            Image:     Rotated image.
            list:      Rotated key points.
        """
        img, data_dict = super().__call__(img, **kwargs)

        rotate_degree = random.uniform(-self.max_degree, self.max_degree)
        height, width, _ = img.shape
        img_center = (width / 2.0, height / 2.0)
        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]

        return self._process(
            img, data_dict,
            random.random() > self.ratio,
            rotate_mat, new_width, new_height
        )


class RandomCrop(_BaseTransform):
    """Crop the given numpy.ndarray and  at a random location.

    Args:
        size (int or tuple): Desired output size of the crop.(w, h)
    """

    def __init__(self, crop_size, crop_ratio=0.5, method='random', grid=None, allow_outside_center=True):
        self.ratio = crop_ratio
        self.method = method
        self.grid = grid
        self.allow_outside_center = allow_outside_center

        if isinstance(crop_size, float):
            self.size = (crop_size, crop_size)
        elif isinstance(crop_size, collections.Iterable) and len(crop_size) == 2:
            self.size = crop_size
        else:
            raise TypeError('Got inappropriate size arg: {}'.format(crop_size))

    def get_lefttop(self, crop_size, img_size):
        if self.method == 'center':
            return [(img_size[0] - crop_size[0]) // 2, (img_size[1] - crop_size[1]) // 2]

        elif self.method == 'random':
            x = random.randint(0, img_size[0] - crop_size[0])
            y = random.randint(0, img_size[1] - crop_size[1])
            return [x, y]

        elif self.method == 'grid':
            grid_x = random.randint(0, self.grid[0] - 1)
            grid_y = random.randint(0, self.grid[1] - 1)
            x = grid_x * ((img_size[0] - crop_size[0]) // (self.grid[0] - 1))
            y = grid_y * ((img_size[1] - crop_size[1]) // (self.grid[1] - 1))
            return [x, y]

        else:
            Log.error('Crop method {} is invalid.'.format(self.method))
            exit(1)

    def _crop(self, x, offset_up, offset_left, target_size):
        return x[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]

    def _process_img(self, img, *args):
        return self._crop(img, *args)

    def _process_labelmap(self, x, *args):
        return self._crop(x, *args)

    def _process_region_indexmap(self, x, *args):
        return self._crop(x, *args)

    def _process_maskmap(self, x, *args):
        return self._crop(x, *args)

    def _process_distance_map(self, x, *args):
        return self._crop(x, *args)

    def _process_angle_map(self, x, *args):
        return self._crop(x, *args)

    def _process_boundary_map(self, x, *args):
        return self._crop(x, *args)

    def _process_multi_label_direction_map(self, x, *args):
        return self._crop(x, *args)

    # def _process_offsetmap_h(self, x, *args):
    #     return self._crop(x, *args)

    # def _process_offsetmap_w(self, x, *args):
    #     return self._crop(x, *args)

    def _process_offsetmap(self, x, *args):
        return self._crop(x, *args)

    def __call__(self, img, **kwargs):
        """
        Args:
            img (Image):   Image to be cropped.
            maskmap (Image):  Mask to be cropped.

        Returns:
            Image:  Cropped image.
            Image:  Cropped maskmap.
            list:   Cropped keypoints.
            list:   Cropped center points.
        """
        img, data_dict = super().__call__(img, **kwargs)

        height, width, _ = img.shape
        target_size = [min(self.size[0], width), min(self.size[1], height)]

        offset_left, offset_up = self.get_lefttop(target_size, [width, height])
        return self._process(
            img, data_dict,
            random.random() > self.ratio,
            offset_up, offset_left, target_size
        )


class Resize(RandomResize):
    """Resize the given numpy.ndarray to random size and aspect ratio.
    Args:
        scale_min: the min scale to resize.
        scale_max: the max scale to resize.
    """

    def __init__(self, target_size=None, min_side_length=None, max_side_length=None, max_side_bound=None):
        self.target_size = target_size
        self.min_side_length = min_side_length
        self.max_side_length = max_side_length
        self.max_side_bound = max_side_bound

    def __call__(self, img, **kwargs):
        img, data_dict = super(RandomResize, self).__call__(img, **kwargs)

        height, width, _ = img.shape
        if self.target_size is not None:
            target_size = self.target_size
            w_scale_ratio = self.target_size[0] / width
            h_scale_ratio = self.target_size[1] / height

        elif self.min_side_length is not None:
            scale_ratio = self.min_side_length / min(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)),
                           int(round(height * h_scale_ratio))]

        else:
            scale_ratio = self.max_side_length / max(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)),
                           int(round(height * h_scale_ratio))]

        if self.max_side_bound is not None and max(target_size) > self.max_side_bound:
            d_ratio = self.max_side_bound / max(target_size)
            w_scale_ratio = d_ratio * w_scale_ratio
            h_scale_ratio = d_ratio * h_scale_ratio
            target_size = [int(round(width * w_scale_ratio)),
                           int(round(height * h_scale_ratio))]

        target_size = tuple(target_size)
        return self._process(
            img, data_dict,
            False,
            target_size, h_scale_ratio, w_scale_ratio
        )


class CV2AugCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> CV2AugCompose([
        >>>     RandomCrop(),
        >>> ])
    """

    def __init__(self, configer, split='train'):
        self.configer = configer
        self.split = split

        if self.split == 'train':
            shuffle_train_trans = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    train_trans_seq_list = self.configer.get(
                        'train_trans', 'shuffle_trans_seq')
                    for train_trans_seq in train_trans_seq_list:
                        shuffle_train_trans += train_trans_seq

                else:
                    shuffle_train_trans = self.configer.get(
                        'train_trans', 'shuffle_trans_seq')
            trans_seq = self.configer.get(
                'train_trans', 'trans_seq') + shuffle_train_trans
            trans_key = 'train_trans'
        else:
            trans_seq = self.configer.get('val_trans', 'trans_seq')
            trans_key = 'val_trans'

        self.transforms = dict()
        self.trans_config = self.configer.get(trans_key)
        for trans_name in trans_seq:
            specs = TRANSFORM_SPEC[trans_name]
            config = self.configer.get(trans_key, trans_name)
            for spec in specs:
                if 'when' not in spec:
                    break
                choose_this = True
                for cond_key, cond_value in spec['when'].items():
                    choose_this = choose_this and (
                        config[cond_key] == cond_value)
                if choose_this:
                    break
            else:
                raise RuntimeError("Not support!")

            kwargs = {}
            for arg_name, arg_path in spec["args"].items():
                if isinstance(arg_path, str):
                    arg_value = config.get(arg_path, None)
                elif isinstance(arg_path, list):
                    arg_value = self.configer.get(*arg_path)
                kwargs[arg_name] = arg_value

            klass = TRANSFORM_MAPPING[trans_name]
            self.transforms[trans_name] = klass(**kwargs)

    def __call__(self, img, **data_dict):

        orig_key_list = list(data_dict)

        if self.configer.get('data', 'input_mode') == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.split == 'train':
            shuffle_trans_seq = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    shuffle_trans_seq_list = self.configer.get('train_trans', 'shuffle_trans_seq')
                    shuffle_trans_seq = shuffle_trans_seq_list[random.randint(0, len(shuffle_trans_seq_list))]
                else:
                    shuffle_trans_seq = self.configer.get('train_trans', 'shuffle_trans_seq')
                    random.shuffle(shuffle_trans_seq)
            trans_seq = shuffle_trans_seq + self.configer.get('train_trans', 'trans_seq')
        else:
            trans_seq = self.configer.get('val_trans', 'trans_seq')

        for trans_key in trans_seq:
            img, data_dict = self.transforms[trans_key](img, **data_dict)

        if self.configer.get('data', 'input_mode') == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return (img, *[data_dict[key] for key in orig_key_list])

    def __repr__(self):
        import pprint
        return 'CV2AugCompose({})'.format(pprint.pformat(self.trans_config))


TRANSFORM_MAPPING = {
    "random_saturation": RandomSaturation,
    "random_hue": RandomHue,
    "random_perm": RandomPerm,
    "random_contrast": RandomContrast,
    "padding": Padding,
    "random_brightness": RandomBrightness,
    "random_hflip": RandomHFlip,
    "random_resize": RandomResize,
    "random_crop": RandomCrop,
    "random_rotate": RandomRotate,
    "resize": Resize,
}

TRANSFORM_SPEC = {
    "random_style": [{
        "args": {
            "style_ratio": "ratio"
        }
    }],
    "random_saturation": [{
        "args": {
            "lower": "lower",
            "upper": "upper",
            "saturation_ratio": "ratio"
        }
    }],
    "random_hue": [{
        "args": {
            "delta": "delta",
            "hue_ratio": "ratio"
        }
    }],
    "ramdom_perm": [{
        "args": {
            "perm_ratio": "ratio"
        }
    }],
    "random_contrast": [{
        "args": {
            "lower": "lower",
            "upper": "upper",
            "contrast_ratio": "ratio"
        }
    }],
    "padding": [{
        "args": {
            "pad": "pad",
            "pad_ratio": "ratio",
            "mean": ["normalize", "mean_value"],
            "allow_outside_center": "allow_outside_center"
        }
    }],
    "random_brightness": [{
        "args": {
            "shift_value": "shift_value",
            "brightness_ratio": "ratio"
        }
    }],
    "random_hflip": [{
        "args": {
            "swap_pair": "swap_pair",
            "flip_ratio": "ratio"
        }
    }],
    "random_resize": [
        {
            "args": {
                "method": "method",
                "scale_range": "scale_range",
                "aspect_range": "aspect_range",
                "max_side_bound": "max_side_bound",
                "resize_ratio": "ratio"
            },
            "when": {
                "method": "random"
            }
        },
        {
            "args": {
                "method": "method",
                "scale_range": "scale_range",
                "aspect_range": "aspect_range",
                "target_size": "target_size",
                "resize_ratio": "ratio"
            },
            "when": {
                "method": "focus"
            }
        },
        {
            "args": {
                "method": "method",
                "aspect_range": "aspect_range",
                "resize_bound": "resize_bound",
                "resize_ratio": "ratio"
            },
            "when": {
                "method": "bound"
            }
        },
    ],
    "random_crop": [
        {
            "args": {
                "crop_size": "crop_size",
                "method": "method",
                "crop_ratio": "ratio",
                "allow_outside_center": "allow_outside_center"
            },
            "when": {
                "method": "random"
            }
        },
        {
            "args": {
                "crop_size": "crop_size",
                "method": "method",
                "crop_ratio": "ratio",
                "allow_outside_center": "allow_outside_center"
            },
            "when": {
                "method": "center"
            }
        },
        {
            "args": {
                "crop_size": "crop_size",
                "method": "method",
                "crop_ratio": "ratio",
                "grid": "grid",
                "allow_outside_center": "allow_outside_center"
            },
            "when": {
                "method": "grid"
            }
        },
    ],
    "random_rotate": [{
        "args": {
            "max_degree": "rotate_degree",
            "rotate_ratio": "ratio",
            "mean": ["normalize", "mean_value"]
        }
    }],
    "resize": [{
        "args": {
            "target_size": "target_size",
            "min_side_length": "min_side_length",
            "max_side_bound": "max_side_bound",
            "max_side_length": "max_side_length"
        }
    }],
}
