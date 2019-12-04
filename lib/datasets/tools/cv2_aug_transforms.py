#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random

import cv2
import numpy as np

from lib.utils.tools.logger import Logger as Log


class Padding(object):
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

    def __call__(self, img, labelmap=None, maskmap=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap

        height, width, channels = img.shape
        left_pad, up_pad, right_pad, down_pad = self.pad

        target_size = [width + left_pad + right_pad, height + up_pad + down_pad]
        offset_left = -left_pad
        offset_up = -up_pad

        expand_image = np.zeros((max(height, target_size[1]) + abs(offset_up),
                                 max(width, target_size[0]) + abs(offset_left), channels), dtype=img.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
        abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = img
        img = expand_image[max(offset_up, 0):max(offset_up, 0) + target_size[1],
              max(offset_left, 0):max(offset_left, 0) + target_size[0]]

        if maskmap is not None:
            expand_maskmap = np.zeros((max(height, target_size[1]) + abs(offset_up),
                                       max(width, target_size[0]) + abs(offset_left)), dtype=maskmap.dtype)
            expand_maskmap[:, :] = 1
            expand_maskmap[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
            abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = maskmap
            maskmap = expand_maskmap[max(offset_up, 0):max(offset_up, 0) + target_size[1],
                      max(offset_left, 0):max(offset_left, 0) + target_size[0]]

        if labelmap is not None:
            expand_labelmap = np.zeros((max(height, target_size[1]) + abs(offset_up),
                                        max(width, target_size[0]) + abs(offset_left)), dtype=labelmap.dtype)
            expand_labelmap[:, :] = 255
            expand_labelmap[abs(min(offset_up, 0)):abs(min(offset_up, 0)) + height,
            abs(min(offset_left, 0)):abs(min(offset_left, 0)) + width] = labelmap
            labelmap = expand_labelmap[max(offset_up, 0):max(offset_up, 0) + target_size[1],
                       max(offset_left, 0):max(offset_left, 0) + target_size[0]]

        return img, labelmap, maskmap


class RandomHFlip(object):
    def __init__(self, swap_pair=None, flip_ratio=0.5):
        self.swap_pair = swap_pair
        self.ratio = flip_ratio

    def __call__(self, img, labelmap=None, maskmap=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap

        height, width, _ = img.shape
        img = cv2.flip(img, 1)
        if labelmap is not None:
            labelmap = cv2.flip(labelmap, 1)
            # to handle datasets with left/right annatations
            if self.swap_pair is not None:
                assert isinstance(self.swap_pair, (tuple, list))
                temp = labelmap.copy()
                for pair in self.swap_pair:
                    assert isinstance(pair, (tuple, list)) and len(pair) == 2
                    labelmap[temp == pair[0]] = pair[1]
                    labelmap[temp == pair[1]] = pair[0]

        if maskmap is not None:
            maskmap = cv2.flip(maskmap, 1)

        return img, labelmap, maskmap


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, saturation_ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = saturation_ratio
        assert self.upper >= self.lower, "saturation upper must be >= lower."
        assert self.lower >= 0, "saturation lower must be non-negative."

    def __call__(self, img, labelmap=None, maskmap=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] *= random.uniform(self.lower, self.upper)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, labelmap, maskmap


class RandomHue(object):
    def __init__(self, delta=18, hue_ratio=0.5):
        assert 0 <= delta <= 360
        self.delta = delta
        self.ratio = hue_ratio

    def __call__(self, img, labelmap=None, maskmap=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap

        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] += random.uniform(-self.delta, self.delta)
        img[:, :, 0][img[:, :, 0] > 360] -= 360
        img[:, :, 0][img[:, :, 0] < 0] += 360
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, labelmap, maskmap


class RandomPerm(object):
    def __init__(self, perm_ratio=0.5):
        self.ratio = perm_ratio
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, labelmap=None, maskmap=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap

        swap = self.perms[random.randint(0, len(self.perms) - 1)]
        img = img[:, :, swap].astype(np.uint8)
        return img, labelmap, maskmap


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, contrast_ratio=0.5):
        self.lower = lower
        self.upper = upper
        self.ratio = contrast_ratio
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, labelmap=None, maskmap=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap

        img = img.astype(np.float32)
        img *= random.uniform(self.lower, self.upper)
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img, labelmap, maskmap


class RandomBrightness(object):
    def __init__(self, shift_value=30, brightness_ratio=0.5):
        self.shift_value = shift_value
        self.ratio = brightness_ratio

    def __call__(self, img, labelmap=None, maskmap=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap

        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img, labelmap, maskmap


class RandomResize(object):
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
                raise TypeError('Got inappropriate size arg: {}'.format(target_size))
        else:
            self.input_size = None

    def get_scale(self, img_size):
        if self.method == 'random':
            scale_ratio = random.uniform(self.scale_range[0], self.scale_range[1])
            return scale_ratio

        elif self.method == 'bound':
            scale1 = self.resize_bound[0] / min(img_size)
            scale2 = self.resize_bound[1] / max(img_size)
            scale = min(scale1, scale2)
            return scale

        else:
            Log.error('Resize method {} is invalid.'.format(self.method))
            exit(1)

    def __call__(self, img, labelmap=None, maskmap=None):
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
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        height, width, _ = img.shape
        if random.random() < self.ratio:
            if self.scale_list is None:
                scale_ratio = self.get_scale([width, height])
            else:
                scale_ratio = self.scale_list[random.randint(0, len(self.scale_list)-1)]

            aspect_ratio = random.uniform(*self.aspect_range)
            w_scale_ratio = math.sqrt(aspect_ratio) * scale_ratio
            h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_ratio
            if self.max_side_bound is not None and max(height*h_scale_ratio, width*w_scale_ratio) > self.max_side_bound:
                d_ratio = self.max_side_bound / max(height * h_scale_ratio, width * w_scale_ratio)
                w_scale_ratio *= d_ratio
                h_scale_ratio *= d_ratio

        else:
            w_scale_ratio, h_scale_ratio = 1.0, 1.0

        converted_size = (int(width * w_scale_ratio), int(height * h_scale_ratio))

        img = cv2.resize(img, converted_size, interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        if labelmap is not None:
            labelmap = cv2.resize(labelmap, converted_size, interpolation=cv2.INTER_NEAREST)

        if maskmap is not None:
            maskmap = cv2.resize(maskmap, converted_size, interpolation=cv2.INTER_NEAREST)

        return img, labelmap, maskmap


class RandomRotate(object):
    """Rotate the input numpy.ndarray and points to the given degree.

    Args:
        degree (number): Desired rotate degree.
    """

    def __init__(self, max_degree, rotate_ratio=0.5, mean=(104, 117, 123)):
        assert isinstance(max_degree, int)
        self.max_degree = max_degree
        self.ratio = rotate_ratio
        self.mean = mean

    def __call__(self, img, labelmap=None, maskmap=None):
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
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() < self.ratio:
            rotate_degree = random.uniform(-self.max_degree, self.max_degree)
        else:
            return img, labelmap, maskmap

        height, width, _ = img.shape

        img_center = (width / 2.0, height / 2.0)

        rotate_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
        cos_val = np.abs(rotate_mat[0, 0])
        sin_val = np.abs(rotate_mat[0, 1])
        new_width = int(height * sin_val + width * cos_val)
        new_height = int(height * cos_val + width * sin_val)
        rotate_mat[0, 2] += (new_width / 2.) - img_center[0]
        rotate_mat[1, 2] += (new_height / 2.) - img_center[1]
        img = cv2.warpAffine(img, rotate_mat, (new_width, new_height), borderValue=self.mean).astype(np.uint8)
        if labelmap is not None:
            labelmap = cv2.warpAffine(labelmap, rotate_mat, (new_width, new_height),
                                      borderValue=(255, 255, 255), flags=cv2.INTER_NEAREST)
            labelmap = labelmap.astype(np.uint8)

        if maskmap is not None:
            maskmap = cv2.warpAffine(maskmap, rotate_mat, (new_width, new_height),
                                     borderValue=(1, 1, 1), flags=cv2.INTER_NEAREST)
            maskmap = maskmap.astype(np.uint8)

        return img, labelmap, maskmap


class RandomCrop(object):
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

    def __call__(self, img, labelmap=None, maskmap=None):
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
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        if random.random() > self.ratio:
            return img, labelmap, maskmap

        height, width, _ = img.shape
        target_size = [min(self.size[0], width), min(self.size[1], height)]

        offset_left, offset_up = self.get_lefttop(target_size, [width, height])

        img = img[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]
        if maskmap is not None:
            maskmap = maskmap[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]

        if labelmap is not None:
            labelmap = labelmap[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]

        return img, labelmap, maskmap


class Resize(object):
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

    def __call__(self, img, labelmap=None, maskmap=None):
        assert isinstance(img, np.ndarray)
        assert labelmap is None or isinstance(labelmap, np.ndarray)
        assert maskmap is None or isinstance(maskmap, np.ndarray)

        height, width, _ = img.shape
        if self.target_size is not None:
            target_size = self.target_size
            w_scale_ratio = self.target_size[0] / width
            h_scale_ratio = self.target_size[1] / height

        elif self.min_side_length is not None:
            scale_ratio = self.min_side_length / min(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        else:
            scale_ratio = self.max_side_length / max(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        if self.max_side_bound is not None and max(target_size) > self.max_side_bound:
            d_ratio = self.max_side_bound / max(target_size)
            w_scale_ratio = d_ratio * w_scale_ratio
            h_scale_ratio = d_ratio * h_scale_ratio
            target_size = [int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio))]

        target_size = tuple(target_size)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        if labelmap is not None:
            labelmap = cv2.resize(labelmap, target_size, interpolation=cv2.INTER_NEAREST)

        if maskmap is not None:
            maskmap = cv2.resize(maskmap, target_size, interpolation=cv2.INTER_NEAREST)

        return img, labelmap, maskmap


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

        self.transforms = dict()
        if self.split == 'train':
            shuffle_train_trans = []
            if self.configer.exists('train_trans', 'shuffle_trans_seq'):
                if isinstance(self.configer.get('train_trans', 'shuffle_trans_seq')[0], list):
                    train_trans_seq_list = self.configer.get('train_trans', 'shuffle_trans_seq')
                    for train_trans_seq in train_trans_seq_list:
                        shuffle_train_trans += train_trans_seq

                else:
                    shuffle_train_trans = self.configer.get('train_trans', 'shuffle_trans_seq')

            if 'random_saturation' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_saturation'] = RandomSaturation(
                    lower=self.configer.get('train_trans', 'random_saturation')['lower'],
                    upper=self.configer.get('train_trans', 'random_saturation')['upper'],
                    saturation_ratio=self.configer.get('train_trans', 'random_saturation')['ratio']
                )

            if 'random_hue' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_hue'] = RandomHue(
                    delta=self.configer.get('train_trans', 'random_hue')['delta'],
                    hue_ratio=self.configer.get('train_trans', 'random_hue')['ratio']
                )

            if 'random_perm' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_perm'] = RandomPerm(
                    perm_ratio=self.configer.get('train_trans', 'random_perm')['ratio']
                )

            if 'random_contrast' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_contrast'] = RandomContrast(
                    lower=self.configer.get('train_trans', 'random_contrast')['lower'],
                    upper=self.configer.get('train_trans', 'random_contrast')['upper'],
                    contrast_ratio=self.configer.get('train_trans', 'random_contrast')['ratio']
                )

            if 'padding' in self.configer.get('train_trans', 'trans_seq'):
                self.transforms['padding'] = Padding(
                    pad=self.configer.get('train_trans', 'padding')['pad'],
                    pad_ratio=self.configer.get('train_trans', 'padding')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value'),
                    allow_outside_center=self.configer.get('train_trans', 'padding')['allow_outside_center']
                )

            if 'random_brightness' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_brightness'] = RandomBrightness(
                    shift_value=self.configer.get('train_trans', 'random_brightness')['shift_value'],
                    brightness_ratio=self.configer.get('train_trans', 'random_brightness')['ratio']
                )

            if 'random_hflip' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_hflip'] = RandomHFlip(
                    swap_pair=self.configer.get('train_trans', 'random_hflip')['swap_pair'],
                    flip_ratio=self.configer.get('train_trans', 'random_hflip')['ratio']
                )

            if 'random_resize' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if self.configer.get('train_trans', 'random_resize')['method'] == 'random':
                    if 'scale_list' not in self.configer.get('train_trans', 'random_resize'):
                        if 'max_side_bound' in self.configer.get('train_trans', 'random_resize'):
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('train_trans', 'random_resize')['method'],
                                scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                                aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                                max_side_bound=self.configer.get('train_trans', 'random_resize')['max_side_bound'],
                                resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                            )
                        else:
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('train_trans', 'random_resize')['method'],
                                scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                                aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                                resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                            )
                    else:
                        if 'max_side_bound' in self.configer.get('train_trans', 'random_resize'):
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('train_trans', 'random_resize')['method'],
                                scale_list=self.configer.get('train_trans', 'random_resize')['scale_list'],
                                aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                                max_side_bound=self.configer.get('train_trans', 'random_resize')['max_side_bound'],
                                resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                            )
                        else:
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('train_trans', 'random_resize')['method'],
                                scale_list=self.configer.get('train_trans', 'random_resize')['scale_list'],
                                aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                                resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                            )

                elif self.configer.get('train_trans', 'random_resize')['method'] == 'focus':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('train_trans', 'random_resize')['method'],
                        scale_range=self.configer.get('train_trans', 'random_resize')['scale_range'],
                        aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                        target_size=self.configer.get('train_trans', 'random_resize')['target_size'],
                        resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                    )

                elif self.configer.get('train_trans', 'random_resize')['method'] == 'bound':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('train_trans', 'random_resize')['method'],
                        aspect_range=self.configer.get('train_trans', 'random_resize')['aspect_range'],
                        resize_bound=self.configer.get('train_trans', 'random_resize')['resize_bound'],
                        resize_ratio=self.configer.get('train_trans', 'random_resize')['ratio']
                    )

                else:
                    Log.error('Not Support Resize Method!')
                    exit(1)

            if 'random_crop' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if self.configer.get('train_trans', 'random_crop')['method'] == 'random':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('train_trans', 'random_crop')['method'] == 'center':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('train_trans', 'random_crop')['method'] == 'grid':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('train_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('train_trans', 'random_crop')['method'],
                        grid=self.configer.get('train_trans', 'random_crop')['grid'],
                        crop_ratio=self.configer.get('train_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('train_trans', 'random_crop')['allow_outside_center']
                    )

                else:
                    Log.error('Not Support Crop Method!')
                    exit(1)

            if 'random_rotate' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                self.transforms['random_rotate'] = RandomRotate(
                    max_degree=self.configer.get('train_trans', 'random_rotate')['rotate_degree'],
                    rotate_ratio=self.configer.get('train_trans', 'random_rotate')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value')
                )

            if 'resize' in self.configer.get('train_trans', 'trans_seq') + shuffle_train_trans:
                if 'target_size' in self.configer.get('train_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        target_size=self.configer.get('train_trans', 'resize')['target_size']
                    )
                if 'min_side_length' in self.configer.get('train_trans', 'resize'):
                    if 'max_side_bound' in self.configer.get('train_trans', 'resize'):
                        self.transforms['resize'] = Resize(
                            min_side_length=self.configer.get('train_trans', 'resize')['min_side_length'],
                            max_side_bound=self.configer.get('train_trans', 'resize')['max_side_bound'],
                        )
                    else:
                        self.transforms['resize'] = Resize(
                            min_side_length=self.configer.get('train_trans', 'resize')['min_side_length']
                        )
                if 'max_side_length' in self.configer.get('train_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        max_side_length=self.configer.get('train_trans', 'resize')['max_side_length']
                    )

        else:
            if 'random_saturation' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_saturation'] = RandomSaturation(
                    lower=self.configer.get('val_trans', 'random_saturation')['lower'],
                    upper=self.configer.get('val_trans', 'random_saturation')['upper'],
                    saturation_ratio=self.configer.get('val_trans', 'random_saturation')['ratio']
                )

            if 'random_hue' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_hue'] = RandomHue(
                    delta=self.configer.get('val_trans', 'random_hue')['delta'],
                    hue_ratio=self.configer.get('val_trans', 'random_hue')['ratio']
                )

            if 'random_perm' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_perm'] = RandomPerm(
                    perm_ratio=self.configer.get('val_trans', 'random_perm')['ratio']
                )

            if 'random_contrast' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_contrast'] = RandomContrast(
                    lower=self.configer.get('val_trans', 'random_contrast')['lower'],
                    upper=self.configer.get('val_trans', 'random_contrast')['upper'],
                    contrast_ratio=self.configer.get('val_trans', 'random_contrast')['ratio']
                )

            if 'padding' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['padding'] = Padding(
                    pad=self.configer.get('val_trans', 'padding')['pad'],
                    pad_ratio=self.configer.get('val_trans', 'padding')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value'),
                    allow_outside_center=self.configer.get('val_trans', 'padding')['allow_outside_center']
                )

            if 'random_brightness' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_brightness'] = RandomBrightness(
                    shift_value=self.configer.get('val_trans', 'random_brightness')['shift_value'],
                    brightness_ratio=self.configer.get('val_trans', 'random_brightness')['ratio']
                )

            if 'random_hflip' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_hflip'] = RandomHFlip(
                    swap_pair=self.configer.get('val_trans', 'random_hflip')['swap_pair'],
                    flip_ratio=self.configer.get('val_trans', 'random_hflip')['ratio']
                )

            if 'random_resize' in self.configer.get('val_trans', 'trans_seq'):
                if self.configer.get('train_trans', 'random_resize')['method'] == 'random':
                    if 'scale_list' not in self.configer.get('val_trans', 'random_resize'):
                        if 'max_side_bound' in self.configer.get('val_trans', 'random_resize'):
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('val_trans', 'random_resize')['method'],
                                scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                                aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                                max_side_bound=self.configer.get('val_trans', 'random_resize')['max_side_bound'],
                                resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                            )
                        else:
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('val_trans', 'random_resize')['method'],
                                scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                                aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                                resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                            )
                    else:
                        if 'max_side_bound' in self.configer.get('val_trans', 'random_resize'):
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('val_trans', 'random_resize')['method'],
                                scale_list=self.configer.get('val_trans', 'random_resize')['scale_list'],
                                aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                                max_side_bound=self.configer.get('val_trans', 'random_resize')['max_side_bound'],
                                resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                            )
                        else:
                            self.transforms['random_resize'] = RandomResize(
                                method=self.configer.get('val_trans', 'random_resize')['method'],
                                scale_list=self.configer.get('val_trans', 'random_resize')['scale_list'],
                                aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                                resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                            )

                elif self.configer.get('val_trans', 'random_resize')['method'] == 'focus':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('val_trans', 'random_resize')['method'],
                        scale_range=self.configer.get('val_trans', 'random_resize')['scale_range'],
                        aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                        target_size=self.configer.get('val_trans', 'random_resize')['target_size'],
                        resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                    )

                elif self.configer.get('val_trans', 'random_resize')['method'] == 'bound':
                    self.transforms['random_resize'] = RandomResize(
                        method=self.configer.get('val_trans', 'random_resize')['method'],
                        aspect_range=self.configer.get('val_trans', 'random_resize')['aspect_range'],
                        resize_bound=self.configer.get('val_trans', 'random_resize')['resize_bound'],
                        resize_ratio=self.configer.get('val_trans', 'random_resize')['ratio']
                    )

                else:
                    Log.error('Not Support Resize Method!')
                    exit(1)

            if 'random_crop' in self.configer.get('val_trans', 'trans_seq'):
                if self.configer.get('val_trans', 'random_crop')['method'] == 'random':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('val_trans', 'random_crop')['method'] == 'center':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                elif self.configer.get('val_trans', 'random_crop')['method'] == 'grid':
                    self.transforms['random_crop'] = RandomCrop(
                        crop_size=self.configer.get('val_trans', 'random_crop')['crop_size'],
                        method=self.configer.get('val_trans', 'random_crop')['method'],
                        grid=self.configer.get('val_trans', 'random_crop')['grid'],
                        crop_ratio=self.configer.get('val_trans', 'random_crop')['ratio'],
                        allow_outside_center=self.configer.get('val_trans', 'random_crop')['allow_outside_center']
                    )

                else:
                    Log.error('Not Support Crop Method!')
                    exit(1)

            if 'random_rotate' in self.configer.get('val_trans', 'trans_seq'):
                self.transforms['random_rotate'] = RandomRotate(
                    max_degree=self.configer.get('val_trans', 'random_rotate')['rotate_degree'],
                    rotate_ratio=self.configer.get('val_trans', 'random_rotate')['ratio'],
                    mean=self.configer.get('normalize', 'mean_value')
                )

            if 'resize' in self.configer.get('val_trans', 'trans_seq'):
                if 'target_size' in self.configer.get('val_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        target_size=self.configer.get('val_trans', 'resize')['target_size']
                    )
                if 'min_side_length' in self.configer.get('val_trans', 'resize'):
                    if 'max_side_bound' in self.configer.get('val_trans', 'resize'):
                        self.transforms['resize'] = Resize(
                            min_side_length=self.configer.get('val_trans', 'resize')['min_side_length'],
                            max_side_bound=self.configer.get('val_trans', 'resize')['max_side_bound'],
                        )
                    else:
                        self.transforms['resize'] = Resize(
                            min_side_length=self.configer.get('val_trans', 'resize')['min_side_length']
                        )
                if 'max_side_length' in self.configer.get('val_trans', 'resize'):
                    self.transforms['resize'] = Resize(
                        max_side_length=self.configer.get('val_trans', 'resize')['max_side_length']
                    )

    def __check_none(self, key_list, value_list):
        for key, value in zip(key_list, value_list):
            if value == 'y' and key is None:
                return False

            if value == 'n' and key is not None:
                return False

        return True

    def __call__(self, img, labelmap=None, maskmap=None):

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

            for trans_key in (shuffle_trans_seq + self.configer.get('train_trans', 'trans_seq')):
                img, labelmap, maskmap = self.transforms[trans_key](img, labelmap, maskmap)

        else:
            for trans_key in self.configer.get('val_trans', 'trans_seq'):
                img, labelmap, maskmap = self.transforms[trans_key](img, labelmap, maskmap)

        if self.configer.get('data', 'input_mode') == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.__check_none([labelmap, maskmap], ['n', 'n']):
            return img

        if self.__check_none([labelmap, maskmap], ['y', 'n']):
            return img, labelmap

        if self.__check_none([labelmap, maskmap], ['n', 'y']):
            return img, maskmap

        if self.__check_none([labelmap, maskmap], ['y', 'y']):
            return img, labelmap, maskmap

        Log.error('Params is not valid.')
        exit(1)
