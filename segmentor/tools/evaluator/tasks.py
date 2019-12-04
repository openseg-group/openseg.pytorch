import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import Counter

from lib.metrics import running_score as rslib
from lib.metrics import F1_running_score as fscore_rslib
from lib.utils.helpers.offset_helper import DTOffsetConfig, DTOffsetHelper
from lib.utils.tools.logger import Logger as Log
from .base import _BaseEvaluator


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SegTask:
    name = 'seg'

    @staticmethod
    def validate_output_spec(spec, spec_counter):
        assert spec_counter['seg'] <= 1

    @staticmethod
    def running_score(spec, configer):
        if 'seg' not in spec:
            return (None, None, None)

        return (
            {'seg': rslib.RunningScore(configer)},
            'seg',
            'miou'
        )

    @staticmethod
    def eval(outputs, meta, running_scores):
        ori_target = meta['ori_target']
        labelmap = np.argmax(outputs['seg'], axis=-1)
        running_scores['seg'].update(labelmap[None], ori_target[None])


class MaskTask:
    name = 'mask'

    @staticmethod
    def validate_output_spec(spec, spec_counter):
        assert spec_counter['mask'] <= 1

    @staticmethod
    def running_score(spec, configer):
        if 'mask' not in spec:
            return (None, None, None)

        return (
            {
                'mask': rslib.RunningScore(
                    configer, num_classes=2, ignore_index=-1
                )
            },
            'mask',
            'acc'
        )

    @staticmethod
    def get_mask_pred(x):
        if x.ndim == 2:
            pred = _sigmoid(x) > 0.5
        else:
            pred = np.argmax(x, axis=-1)

        return pred.astype(np.int)

    @staticmethod
    def eval(outputs, meta, running_scores):
        distance_map = meta['ori_distance_map']
        seg_label_map = meta['ori_target']
        gt_mask_label = DTOffsetHelper.distance_to_mask_label(
            distance_map,
            seg_label_map,
            return_tensor=False
        )
        mask_pred = MaskTask.get_mask_pred(outputs['mask'])
        running_scores['mask'].update(
            (mask_pred == 1).astype(np.int)[None],
            gt_mask_label[None]
        )


class DirectionTask:
    name = 'dir'

    @staticmethod
    def validate_output_spec(spec, spec_counter):
        assert spec_counter['dir'] == 0 or (
            spec_counter['dir'] == 1 and spec_counter['mask'] == 1
        )

    @staticmethod
    def running_score(spec, configer):
        if 'dir' not in spec:
            return (None, None, None)

        return (
            {
                'dir (mask)': rslib.RunningScore(configer, num_classes=DTOffsetConfig.num_classes, ignore_index=-1),
                'dir (GT)': rslib.RunningScore(configer, num_classes=DTOffsetConfig.num_classes + 1, ignore_index=-1),
            },
            'dir (GT)',
            'acc'
        )

    @staticmethod
    def eval(outputs, meta, running_scores):
        distance_map = meta['ori_distance_map']
        angle_map = meta['ori_angle_map']
        seg_label_map = meta['ori_target']

        mask_pred = MaskTask.get_mask_pred(outputs['mask'])
        dir_pred = np.argmax(outputs['dir'], axis=-1)

        gt_mask_label = DTOffsetHelper.distance_to_mask_label(
            distance_map,
            seg_label_map,
            return_tensor=False
        )
        gt_dir_label = DTOffsetHelper.angle_to_direction_label(
            angle_map,
            seg_label_map=seg_label_map,
            extra_ignore_mask=mask_pred != 1
        )

        running_scores['dir (mask)'].update(
            dir_pred[None], gt_dir_label[None]
        )

        dir_pred[mask_pred != 1] = DTOffsetConfig.num_classes
        gt_dir_label = DTOffsetHelper.angle_to_direction_label(
            angle_map,
            seg_label_map=seg_label_map,
            extra_ignore_mask=(gt_mask_label == 0),
        )
        running_scores['dir (GT)'].update(
            dir_pred[None], gt_dir_label[None]
        )


class MLDirectionTask:
    name = 'ml_dir'

    @staticmethod
    def validate_output_spec(spec, spec_counter):
        assert spec_counter['ml_dir'] == 0 or (
            spec_counter['ml_dir'] == 1 and spec_counter['mask'] == 1
        )

    @staticmethod
    def running_score(spec, configer):
        if 'ml_dir' not in spec:
            return (None, None, None)

        return (
            {
                'ML dir (mask)': rslib.MultiLabelRunningScore(),
                'ML dir (GT)': rslib.MultiLabelRunningScore(),
            },
            'ML dir (GT)',
            'acc'
        )

    @staticmethod
    def _get_multilabel_prediction(dir_logits, no_offset_mask=None, topk=8):
        h, w, _ = dir_logits.shape
        dir_logits = torch.from_numpy(
            dir_logits
        ).unsqueeze(0).permute(0, 3, 1, 2)
        offsets = []
        if topk == dir_logits.shape[1]:
            for i in range(topk):
                offset_i = DTOffsetHelper.label_to_vector(
                    torch.tensor([i]).view(1, 1, 1)
                ).repeat(1, 1, h, w)
                offset_i = offset_i.float() * dir_logits[:, i:i+1, :, :]
                offsets.append(offset_i)
        else:
            dir_logits, dir_pred = torch.topk(dir_logits, topk, dim=1)
            for i in range(topk):
                dir_pred_i = dir_pred[:, i, :, :]
                offset_i = DTOffsetHelper.label_to_vector(dir_pred_i)
                offset_i = offset_i.float() * dir_logits[:, i:i+1, :, :]
                offsets.append(offset_i)

        offset = sum(offsets)
        dir_pred = DTOffsetHelper.vector_to_label(
            offset.permute(0, 2, 3, 1),
            num_classes=8,
            return_tensor=True
        )

        dir_pred = dir_pred.squeeze(0).numpy()

        if no_offset_mask is not None:
            dir_pred[no_offset_mask] = 8

        return dir_pred

    @staticmethod
    def eval(outputs, meta, running_scores):
        distance_map = meta['ori_distance_map']
        seg_label_map = meta['ori_target']
        dir_label_map = meta['ori_multi_label_direction_map']
        dir_label_map = DTOffsetHelper.encode_multi_labels(dir_label_map)
        dir_label_map[seg_label_map == -1, :] = -1
        gt_mask_label = DTOffsetHelper.distance_to_mask_label(
            distance_map,
            seg_label_map,
            return_tensor=False
        )

        mask_pred = MaskTask.get_mask_pred(outputs['mask'])
        dir_pred = MLDirectionTask._get_multilabel_prediction(
            outputs['ml_dir'],
            no_offset_mask=mask_pred == 0,
            topk=8
        )

        running_scores['ML dir (mask)'].update(
            dir_pred, dir_label_map,
            (mask_pred == 1) & (seg_label_map != -1)
        )
        running_scores['ML dir (GT)'].update(
            dir_pred, dir_label_map,
            gt_mask_label == 1
        )


task_mapping = {task.name: task for task in [
    MaskTask,
    SegTask,
    DirectionTask,
    MLDirectionTask,
]}
