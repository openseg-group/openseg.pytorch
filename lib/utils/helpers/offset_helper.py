##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: JingyiXie
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import torch
import numpy as np
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log

ori_scales = {
    4: 1,
    8: 1,
    16: 2,
    32: 4,
}


class DTOffsetConfig:
    # energy configurations
    energy_level_step = int(os.environ.get('dt_energy_level_step', 5))
    assert energy_level_step > 0

    max_distance = int(os.environ.get('dt_max_distance', 5))
    min_distance = int(os.environ.get('dt_min_distance', 0))

    num_energy_levels = max_distance // energy_level_step + 1

    offset_min_level = int(os.environ.get('dt_offset_min_level', 0))
    offset_max_level = int(os.environ.get('dt_offset_max_level', 5))
    # assert 0 <= offset_min_level < num_energy_levels - 1
    # assert 0 < offset_max_level <= num_energy_levels

    # direction configurations
    num_classes = int(os.environ.get('dt_num_classes', 8))
    assert num_classes in (4, 8, 16, 32,)

    # offset scale configurations
    scale = int(os.environ.get('dt_scale', ori_scales[num_classes]))
    assert scale % ori_scales[num_classes] == 0
    scale //= ori_scales[num_classes]

    c4_align_axis = os.environ.get('c4_align_axis') is not None

    Log.info(
        'engery/max-distance: {} engery/min-distance: {}'.format(
            max_distance,
            min_distance
        )
    )

    Log.info(
        'direction/num_classes: {} scale: {}'.format(
            num_classes,
            scale
        )
    )

    Log.info(
        'c4 align axis: {}'.format(c4_align_axis)
    )


label_to_vector_mapping = {
    4: [
        [-1, -1], [-1, 1], [1, 1], [1, -1]
    ] if not DTOffsetConfig.c4_align_axis else [
        [0, -1], [-1, 0], [0, 1], [1, 0]
    ],    
    8: [
        [0, -1], [-1, -1], [-1, 0], [-1, 1],
        [0, 1], [1, 1], [1, 0], [1, -1]
    ],
    16: [
        [0, -2], [-1, -2], [-2, -2], [-2, -1], 
        [-2, 0], [-2, 1], [-2, 2], [-1, 2],
        [0, 2], [1, 2], [2, 2], [2, 1],
        [2, 0], [2, -1], [2, -2], [1, -2]
    ],
    32: [
        [0, -4], [-1, -4], [-2, -4], [-3, -4], [-4, -4], [-4, -3], [-4, -2], [-4, -1],
        [-4, 0], [-4, 1], [-4, 2], [-4, 3], [-4, 4], [-3, 4], [-2, 4], [-1, 4],
        [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [4, 3], [4, 2], [4, 1],
        [4, 0], [4, -1], [4, -2], [4, -3], [4, -4], [3, -4], [2, -4], [1, -4],
    ]
}

vector_to_label_mapping = {
    8: list(range(8)),
    16: list(range(16)),
}


class Sobel:

    _caches = {}
    ksize = 11

    @staticmethod
    def _generate_sobel_kernel(shape, axis):
        """
        shape must be odd: eg. (5,5)
        axis is the direction, with 0 to positive x and 1 to positive y
        """
        k = np.zeros(shape, dtype=np.float32)
        p = [
            (j, i)
            for j in range(shape[0])
            for i in range(shape[1])
            if not (i == (shape[1] - 1) / 2.0 and j == (shape[0] - 1) / 2.0)
        ]

        for j, i in p:
            j_ = int(j - (shape[0] - 1) / 2.0)
            i_ = int(i - (shape[1] - 1) / 2.0)
            k[j, i] = (i_ if axis == 0 else j_) / float(i_ * i_ + j_ * j_)
        return torch.from_numpy(k).unsqueeze(0)

    @classmethod
    def kernel(cls, ksize=None):
        if ksize is None:
            ksize = cls.ksize
        if ksize in cls._caches:
            return cls._caches[ksize]

        sobel_x, sobel_y = (cls._generate_sobel_kernel((ksize, ksize), i) for i in (0, 1))
        sobel_ker = torch.cat([sobel_y, sobel_x], dim=0).view(2, 1, ksize, ksize)
        cls._caches[ksize] = sobel_ker
        return sobel_ker


class DTOffsetHelper:

    @staticmethod
    def encode_multi_labels(dir_labels):
        """
        Only accept ndarray of shape H x W (uint8).
        """
        assert isinstance(dir_labels, np.ndarray)

        output = np.zeros((*dir_labels.shape, 8), dtype=np.int)
        for i in range(8):
            output[..., i] = (dir_labels & (1 << i) != 0).astype(np.int)

        return output

    @staticmethod
    def edge_mask_to_vector(edge_mask, kernel_size=Sobel.ksize, normalized=True):
        """
        `edge_mask` -> 1 indicates edge.
        """
        edge_mask = torch.clamp(edge_mask, min=0, max=1)
        edge_mask = 1 - edge_mask

        sobel_kernel = Sobel.kernel(ksize=kernel_size).to(edge_mask.device)
        direction = F.conv2d(
            edge_mask,
            sobel_kernel,
            padding=kernel_size // 2
        )

        if normalized:
            direction = F.normalize(direction, dim=1)

        return direction

    @staticmethod
    def binary_mask_map_to_offset(bmap):
        """
        refer to: https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/41065243#41065243
        apply sobel on the binary edge map to estimate the offset directions for the edge pixels.
        """
        from scipy.ndimage.morphology import distance_transform_edt

        depths = []
        _, h, w = bmap.size()
        for bmap_i in (1 - bmap).cpu().numpy():
            depth_i = distance_transform_edt(bmap_i)            
            depths.append(torch.from_numpy(depth_i).view(1, 1, h, w))

        depths = torch.cat(depths, dim=0).to(bmap.device)
        offsets = F.conv2d(depths, Sobel.kernel().to(bmap.device), padding=Sobel.ksize // 2)
        angles = torch.atan2(offsets[:, 0], offsets[:, 1]) / np.pi * 180
        offset = DTOffsetHelper.angle_to_offset(angles, return_tensor=True)
        offset[(bmap == 1).unsqueeze(-1).repeat(1, 1, 1, 2)] = 0
        return offset

    @staticmethod
    def distance_to_energy_label(distance_map, 
                                 seg_label_map, 
                                 return_tensor=False):
        if return_tensor:
            assert isinstance(distance_map, torch.Tensor)
            assert isinstance(seg_label_map, torch.Tensor)
        else:
            assert isinstance(distance_map, np.ndarray)
            assert isinstance(seg_label_map, np.ndarray)

        if return_tensor:
            energy_label_map = torch.zeros_like(seg_label_map).long().to(distance_map.device)
        else:
            energy_label_map = np.zeros(seg_label_map.shape, dtype=np.int)

        keep_mask = seg_label_map != -1
        energy_level_step = DTOffsetConfig.energy_level_step

        for i in range(DTOffsetConfig.num_energy_levels - 1):
            energy_label_map[keep_mask & (
                distance_map >= i * energy_level_step) & (distance_map < (i + 1) * energy_level_step)] = i
        
        energy_label_map[keep_mask & (
            distance_map >= DTOffsetConfig.max_distance)] = DTOffsetConfig.num_energy_levels - 1

        energy_label_map[~keep_mask] = -1

        return energy_label_map

    @staticmethod
    def logits_to_vector(dir_map):
        dir_map = F.softmax(dir_map, dim=1)

        n, _, h, w = dir_map.shape
        offsets = DTOffsetHelper.label_to_vector(
            torch.arange(DTOffsetConfig.num_classes).view(DTOffsetConfig.num_classes, 1, 1).cuda()
        ).float().unsqueeze(0)  # 1 x 8 x 2 x 1 x 1
        offsets_h = offsets[:, :, 0].repeat(n, 1, h, w)  # n x 8 x h x w
        offsets_w = offsets[:, :, 1].repeat(n, 1, h, w)  # n x 8 x h x w
        offsets = torch.stack([
            (offsets_h * dir_map).sum(dim=1),
            (offsets_w * dir_map).sum(dim=1),
        ], dim=1)
        offsets = F.normalize(offsets, p=2, dim=1)

        return offsets

    @staticmethod
    def get_opposite_angle(angle_map):
        new_angle_map = angle_map + 180
        mask = (new_angle_map >= 180) & (new_angle_map <= 360)
        new_angle_map[mask] = new_angle_map[mask] - 360
        return new_angle_map

    @staticmethod
    def label_to_vector(labelmap, 
                        num_classes=DTOffsetConfig.num_classes):

        assert isinstance(labelmap, torch.Tensor)

        mapping = label_to_vector_mapping[num_classes]
        offset_h = torch.zeros_like(labelmap).long()
        offset_w = torch.zeros_like(labelmap).long()

        for idx, (hdir, wdir) in enumerate(mapping):
            mask = labelmap == idx
            offset_h[mask] = hdir
            offset_w[mask] = wdir

        return torch.stack([offset_h, offset_w], dim=-1).permute(0, 3, 1, 2).to(labelmap.device)

    @staticmethod
    def distance_to_mask_label(distance_map, 
                               seg_label_map, 
                               return_tensor=False):

        if return_tensor:
            assert isinstance(distance_map, torch.Tensor)
            assert isinstance(seg_label_map, torch.Tensor)
        else:
            assert isinstance(distance_map, np.ndarray)
            assert isinstance(seg_label_map, np.ndarray)

        if return_tensor:
            mask_label_map = torch.zeros_like(seg_label_map).long().to(distance_map.device)
        else:
            mask_label_map = np.zeros(seg_label_map.shape, dtype=np.int)

        keep_mask = (distance_map <= DTOffsetConfig.max_distance) & (distance_map >= DTOffsetConfig.min_distance)
        mask_label_map[keep_mask] = 1
        mask_label_map[seg_label_map == -1] = -1

        return mask_label_map

    @staticmethod
    def align_angle_c4(angle_map, return_tensor=False):
        """
        [-180, -90) -> 0
        [-90, 0) -> 1
        [0, 90) -> 2
        [90, 180) -> 3
        """

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
        else:
            angle_map = torch.from_numpy(angle_map)

        angle_index_map = torch.trunc((angle_map + 180) / 90).long()
        angle_index_map = torch.clamp(angle_index_map, min=0, max=3)

        new_angle_map = (angle_index_map * 90 - 135).float()

        if not return_tensor:
            new_angle_map = new_angle_map.numpy()
            angle_index_map = angle_index_map.numpy()

        return new_angle_map, angle_index_map

    @staticmethod
    def align_angle(angle_map, 
                    num_classes=DTOffsetConfig.num_classes, 
                    return_tensor=False):

        if num_classes == 4 and not DTOffsetConfig.c4_align_axis:
            return DTOffsetHelper.align_angle_c4(angle_map, return_tensor=return_tensor)

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
        else:
            assert isinstance(angle_map, np.ndarray)

        step = 360 / num_classes
        if return_tensor:
            new_angle_map = torch.zeros(angle_map.shape).float().to(angle_map.device)
            angle_index_map = torch.zeros(angle_map.shape).long().to(angle_map.device)
        else:
            new_angle_map = np.zeros(angle_map.shape, dtype=np.float)
            angle_index_map = np.zeros(angle_map.shape, dtype=np.int)
        mask = (angle_map <= (-180 + step/2)) | (angle_map > (180 - step/2))
        new_angle_map[mask] = -180
        angle_index_map[mask] = 0

        for i in range(1, num_classes):
            middle = -180 + step * i
            mask = (angle_map > (middle - step / 2)) & (angle_map <= (middle + step / 2))
            new_angle_map[mask] = middle
            angle_index_map[mask] = i

        return new_angle_map, angle_index_map


    @staticmethod
    def angle_to_offset(angle_map, 
                        distance_map=None, 
                        num_classes=DTOffsetConfig.num_classes, 
                        return_tensor=False, 
                        use_scale=False):

        if return_tensor:
            assert isinstance(distance_map, torch.Tensor) or distance_map is None
            assert isinstance(angle_map, torch.Tensor)
        else:
            assert isinstance(distance_map, np.ndarray) or distance_map is None
            assert isinstance(angle_map, np.ndarray)

        _, angle_index_map = DTOffsetHelper.align_angle(
            angle_map, num_classes=num_classes, return_tensor=return_tensor)
        mapping = label_to_vector_mapping[num_classes]

        if use_scale:
            scale = DTOffsetConfig.scale
        else:
            scale = 1

        if distance_map is not None:
            no_offset_mask = (
                (distance_map > DTOffsetConfig.max_distance) |
                (distance_map < DTOffsetConfig.min_distance)
            )
        else:
            no_offset_mask = torch.zeros(angle_map.shape, dtype=torch.uint8).to(angle_map.device)            

        if return_tensor:
            offset_h = torch.zeros(angle_map.shape).long().to(angle_map.device)
            offset_w = torch.zeros(angle_map.shape).long().to(angle_map.device)
        else:
            offset_h = np.zeros(angle_map.shape, dtype=np.int)
            offset_w = np.zeros(angle_map.shape, dtype=np.int)

        for i in range(num_classes):
            mask = (angle_index_map == i) & ~no_offset_mask
            offset_h[mask] = mapping[i][0] * scale
            offset_w[mask] = mapping[i][1] * scale

        if return_tensor:
            return torch.stack([offset_h, offset_w], dim=-1)
        else:
            return np.stack([offset_h, offset_w], axis=-1)


    @staticmethod
    def _vis_offset(_offset, 
                    image_name=None, 
                    image=None, 
                    color=(0, 0, 255), 
                    only_points=False):
        import cv2
        import random
        import os.path as osp
        if image is None:
            color = 255
            image = np.zeros_like(_offset[:, :, 0], dtype=np.uint8)

        if only_points:
            image[(_offset[:, :, 0] != 0) | (_offset[:, :, 1] != 0)] = 255
        else:
            step = 6
            coord_map = torch.stack(torch.meshgrid([torch.arange(
                length) for length in _offset.shape[:-1]]), dim=-1).numpy().astype(np.int)
            offset = (_offset * 10 + coord_map).astype(np.int)
            for i in range(step//2, offset.shape[0], step):
                for j in range(step//2, offset.shape[1], step):
                    if (_offset[i, j] == 0).all():
                        continue
                    cv2.arrowedLine(img=image, pt1=tuple(
                        coord_map[i, j][::-1]), pt2=tuple(offset[i, j][::-1]), color=color, thickness=1)
        if image_name is None:
            image_name = '{}.png'.format(random.random())
        cv2.imwrite('/msravcshare/v-jinxi/vis/{}.png'.format(image_name), image)         

    @staticmethod
    def angle_to_vector(angle_map, 
                        num_classes=DTOffsetConfig.num_classes, 
                        return_tensor=False):

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
        else:
            assert isinstance(angle_map, np.ndarray)

        if return_tensor:
            lib = torch
            vector_map = torch.zeros((*angle_map.shape, 2), dtype=torch.float).to(angle_map.device)
            deg2rad = lambda x: np.pi / 180.0 * x
        else:
            lib = np
            vector_map = np.zeros((*angle_map.shape, 2), dtype=np.float)
            deg2rad = np.deg2rad

        if num_classes is not None:
            angle_map, _ = DTOffsetHelper.align_angle(angle_map, num_classes=num_classes, return_tensor=return_tensor)

        angle_map = deg2rad(angle_map)
        
        vector_map[..., 0] = lib.sin(angle_map)
        vector_map[..., 1] = lib.cos(angle_map)

        return vector_map

    @staticmethod
    def angle_to_direction_label(angle_map, 
                                 seg_label_map=None, 
                                 distance_map=None, 
                                 num_classes=DTOffsetConfig.num_classes, 
                                 extra_ignore_mask=None, 
                                 return_tensor=False):

        if return_tensor:
            assert isinstance(angle_map, torch.Tensor)
            assert isinstance(seg_label_map, torch.Tensor) or seg_label_map is None
        else:
            assert isinstance(angle_map, np.ndarray)
            assert isinstance(seg_label_map, np.ndarray) or seg_label_map is None

        _, label_map = DTOffsetHelper.align_angle(angle_map, 
                                                  num_classes=num_classes, 
                                                  return_tensor=return_tensor)
        if distance_map is not None:
            label_map[distance_map > DTOffsetConfig.max_distance] = num_classes
        if seg_label_map is None:
            if return_tensor:
                ignore_mask = torch.zeros(angle_map.shape, dtype=torch.uint8).to(angle_map.device)
            else:
                ignore_mask = np.zeros(angle_map.shape, dtype=np.bool)
        else:
            ignore_mask = seg_label_map == -1
            
        if extra_ignore_mask is not None:
            ignore_mask = ignore_mask | extra_ignore_mask
        label_map[ignore_mask] = -1

        return label_map

    @staticmethod
    def vector_to_label(vector_map, 
                        num_classes=DTOffsetConfig.num_classes, 
                        return_tensor=False):

        if return_tensor:
            assert isinstance(vector_map, torch.Tensor)
        else:
            assert isinstance(vector_map, np.ndarray)

        if return_tensor:
            rad2deg = lambda x: x * 180. / np.pi
        else:
            rad2deg = np.rad2deg

        angle_map = np.arctan2(vector_map[..., 0], vector_map[..., 1])
        angle_map = rad2deg(angle_map)

        return DTOffsetHelper.angle_to_direction_label(angle_map, 
                                                       return_tensor=return_tensor, 
                                                       num_classes=num_classes)

if __name__ == '__main__':
    angle = torch.tensor([[0., 45., 90., 180., -180.]])
    print(DTOffsetHelper.align_angle(angle, num_classes=4, return_tensor=True))
    raise RuntimeError
    distance_map = torch.tensor([[1., 2., 3., 255., 4.]])
    seg_map = torch.tensor([[-1, 0, 0, 0, 0]])
    print(angle)
    print(DTOffsetHelper.angle_to_direction_label(angle, return_tensor=True, distance_map=distance_map, seg_label_map=seg_map))
    print(DTOffsetHelper.angle_to_offset(angle, return_tensor=True, distance_map=distance_map))
    print(DTOffsetHelper.distance_to_mask_label(distance_map, seg_map, return_tensor=True))
    vector = DTOffsetHelper.angle_to_vector(angle, return_tensor=True)
    print(vector)
    print(DTOffsetHelper.vector_to_label(vector, return_tensor=True))
    angle = np.array([0., 45., 90., 180., -180.])
    distance_map = np.array([1., 2., 3., 255., 4.])
    print(angle)
    print(DTOffsetHelper.angle_to_direction_label(angle, return_tensor=False, distance_map=distance_map))    
    vector = (DTOffsetHelper.angle_to_vector(angle, return_tensor=False))
    print(vector)
    print(DTOffsetHelper.vector_to_label(vector, return_tensor=False))