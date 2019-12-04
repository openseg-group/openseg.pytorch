##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: JingyiXie, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## Code adapted from:
## https://github.com/nv-tlabs/GSCNN/blob/master/utils/f_boundary.py
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import torch
from multiprocessing.pool import Pool


class F1RunningScore(object):

    def __init__(self, configer=None, num_classes=None, boundary_threshold=0.00088, num_proc=15):

        assert configer is not None or num_classes is not None
        self.configer = configer

        if configer is not None:
            self.n_classes = self.configer.get('data', 'num_classes')
        else:
            self.n_classes = num_classes

        self.ignore_index = -1
        self.boundary_threshold = boundary_threshold
        self.pool = Pool(processes=num_proc)
        self.num_proc = num_proc

        self._Fpc = 0
        self._Fc = 0
        self.seg_map_cache = []
        self.gt_map_cache = []

    def _update_cache(self, seg_map, gt_map):
        """
        Append inputs to `seg_map_cache` and `gt_map_cache`.

        Returns whether the length reached our pool size.
        """
        self.seg_map_cache.extend(seg_map)
        self.gt_map_cache.extend(gt_map)
        return len(self.gt_map_cache) >= self.num_proc

    def _get_from_cache(self):

        n = self.num_proc
        seg_map, self.seg_map_cache = self.seg_map_cache[:n], self.seg_map_cache[n:]
        gt_map, self.gt_map_cache = self.gt_map_cache[:n], self.gt_map_cache[n:]

        return seg_map, gt_map

    def update(self, seg_map, gt_map):

        if self._update_cache(seg_map, gt_map):
            seg_map, gt_map = self._get_from_cache()
            self._update_scores(seg_map, gt_map)
        else:
            return

    def _update_scores(self, seg_map, gt_map):
        batch_size = len(seg_map)
        if batch_size == 0:
            return

        Fpc = np.zeros(self.n_classes)
        Fc = np.zeros(self.n_classes)


        for class_id in range(self.n_classes):
            args = []
            for i in range(batch_size):
                if seg_map[i].shape[0] == self.n_classes:
                    pred_i = seg_map[i][class_id] > 0.5
                    pred_is_boundary = True
                else:
                    pred_i = seg_map[i] == class_id
                    pred_is_boundary = False

                args.append([
                    (pred_i).astype(np.uint8),
                    (gt_map[i] == class_id).astype(np.uint8),
                    (gt_map[i] == -1),
                    self.boundary_threshold,
                    class_id,
                    pred_is_boundary
                ])
            results = self.pool.map(db_eval_boundary, args)
            results = np.array(results)
            Fs = results[:, 0]
            _valid = ~np.isnan(Fs)
            Fc[class_id] = np.sum(_valid)
            Fs[np.isnan(Fs)] = 0
            Fpc[class_id] = sum(Fs)

        self._Fc = self._Fc + Fc
        self._Fpc = self._Fpc + Fpc

    def get_scores(self):

        if self.seg_map_cache is None:
            return 0, 0

        self._update_scores(self.seg_map_cache, self.gt_map_cache)

        F_score = np.sum(self._Fpc / self._Fc) / self.n_classes
        F_score_classwise = self._Fpc / self._Fc

        return F_score, F_score_classwise

    def reset(self):
        self._Fpc = self._Fc = 0


def db_eval_boundary(args):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask         (ndarray): binary annotated image.

    Returns:
            F (float): boundaries F-measure
            P (float): boundaries precision
            R (float): boundaries recall
    """

    foreground_mask, gt_mask, ignore_mask, bound_th, class_id, pred_is_boundary = args

    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # print(bound_pix)
    # print(gt.shape)
    # print(np.unique(gt))
    foreground_mask[ignore_mask] = 0
    gt_mask[ignore_mask] = 0

    # Get the pixel boundaries of both masks
    if pred_is_boundary:
        fg_boundary = foreground_mask
    else:
        fg_boundary = seg2bmap(foreground_mask)
    gt_boundary = seg2bmap(gt_mask)

    from skimage.morphology import disk
    from cv2 import dilate
    def binary_dilation(x, d): return dilate(
        x.astype(np.uint8), d).astype(np.bool)
    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F, precision


def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

    Arguments:
            seg     : Segments labeled from 1..k.
            width	:	Width of desired bmap  <= seg.shape[1]
            height  :	Height of desired bmap <= seg.shape[0]

    Returns:
            bmap (ndarray):	Binary boundary map.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01),\
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + floor((y - 1) + height / h)
                    i = 1 + floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap