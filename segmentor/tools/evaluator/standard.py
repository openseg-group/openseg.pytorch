import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import Counter

from lib.utils.helpers.offset_helper import DTOffsetConfig, DTOffsetHelper
from lib.utils.tools.logger import Logger as Log
from .base import _BaseEvaluator
from . import tasks

def _parse_output_spec(spec):
    """
    Parse string like "mask, _, dir, ..., seg" into indices mapping
    {
        "mask": 0,
        "dir": 2,
        "seg": -1
    }
    """
    spec = [x.strip() for x in spec.split(',')]
    existing_task_names = set(tasks.task_mapping)

    # `spec` should not have invalid keys other than in `existing_task_names`
    assert set(spec) - ({'...', '_'} | existing_task_names) == set()
    # `spec` should have at least one key in `existing_task_names`
    assert set(spec) & existing_task_names != set()

    counter = Counter(spec)
    for task in tasks.task_mapping.values():
        task.validate_output_spec(spec, counter)
    assert counter['...'] <= 1

    length = len(spec)
    output_indices = {}
    negative_index = False
    for idx, name in enumerate(spec):
        if name not in ['_', '...']:
            index = idx - length if negative_index else idx
            output_indices[name] = index
        elif name == '...':
            negative_index = True

    return output_indices


class StandardEvaluator(_BaseEvaluator):

    def _output_spec(self):
        if self.configer.conditions.pred_dt_offset:
            default_spec = 'mask, dir'
        elif self.configer.conditions.pred_ml_dt_offset:
            default_spec = 'mask, ml_dir'
        else:
            default_spec = '..., seg'

        return os.environ.get('output_spec', default_spec)

    def _init_running_scores(self):
        self.output_indices = _parse_output_spec(self._output_spec())

        self.running_scores = {}
        for task in tasks.task_mapping.values():
            rss, main_key, metric = task.running_score(self.output_indices, self.configer)
            if rss is None:
                continue
            self.running_scores.update(rss)
            self.save_net_main_key = main_key
            self.save_net_metric = metric

    def update_score(self, outputs, metas):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
            
        for i in range(len(outputs[0])):

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']

            outputs_numpy = {}
            for name, idx in self.output_indices.items():
                item = outputs[idx].permute(0, 2, 3, 1)
                item = cv2.resize(
                    item[i, :border_size[1], :border_size[0]].cpu().numpy(),
                    tuple(ori_img_size), interpolation=cv2.INTER_CUBIC
                )
                outputs_numpy[name] = item

            for name in outputs_numpy:
                tasks.task_mapping[name].eval(
                    outputs_numpy, metas[i], self.running_scores
                )
