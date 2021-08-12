##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: JingyiXie
## Microsoft Research
## hsfzxjy@gmail.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import timeit
import pdb
import cv2
import scipy
import pprint
import collections

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from lib.utils.helpers.file_helper import FileHelper
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.metrics.running_score import RunningScore
from segmentor.tools.module_runner import ModuleRunner
from scipy import ndimage
from PIL import Image
from math import ceil

from lib.utils.helpers.offset_helper import DTOffsetConfig, DTOffsetHelper


class Tester(object):
    def __init__(self, configer):
        self.crop_size = configer.get("train", "data_transformer")["input_size"][::-1]
        val_trans_seq = [
            x for x in configer.get("val_trans", "trans_seq") if "random" not in x
        ]
        configer.update(("val_trans", "trans_seq"), val_trans_seq)
        configer.get("val", "data_transformer")["input_size"] = configer.get(
            "test", "data_transformer"
        ).get("input_size", None)
        configer.update(
            ("train", "data_transformer"), configer.get("val", "data_transformer")
        )
        configer.update(("val", "batch_size"), int(os.environ.get("batch_size", 16)))
        configer.update(("test", "batch_size"), int(os.environ.get("batch_size", 16)))

        self.save_dir = configer.get("test", "out_dir")
        self.dataset_name = configer.get("test", "eval_set")
        self.sscrop = configer.get("test", "sscrop")

        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.seg_data_loader = DataLoader(configer)
        self.seg_net = None
        self.test_loader = None
        self.test_size = None
        self.infer_time = 0
        self.infer_cnt = 0
        self._init_model()

        pprint.pprint(configer.params_root)

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        assert self.dataset_name in (
            "train",
            "val",
            "test",
        ), "Cannot infer dataset name"

        self.size_mode = self.configer.get(self.dataset_name, "data_transformer")[
            "size_mode"
        ]

        if self.dataset_name != "test":
            self.test_loader = self.seg_data_loader.get_valloader(self.dataset_name)
        else:
            self.test_loader = self.seg_data_loader.get_testloader(self.dataset_name)
        self.test_size = len(self.test_loader) * self.configer.get("val", "batch_size")

    def test(self, data_loader=None):
        """
        Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()
        image_id = 0

        Log.info("save dir {}".format(self.save_dir))
        FileHelper.make_dirs(self.save_dir, is_file=False)

        print("Total batches", len(self.test_loader))
        for j, data_dict in enumerate(self.test_loader):
            inputs = [data_dict["img"]]
            names = data_dict["name"]
            metas = data_dict["meta"]

            dest_dir = self.save_dir

            with torch.no_grad():
                offsets, logits = self.extract_offset(inputs)
                print([x.shape for x in logits])
                for k in range(len(inputs[0])):
                    image_id += 1
                    ori_img_size = metas[k]["ori_img_size"]
                    border_size = metas[k]["border_size"]
                    offset = offsets[k].squeeze().cpu().numpy()
                    offset = cv2.resize(
                        offset[: border_size[1], : border_size[0]],
                        tuple(ori_img_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    print(image_id)

                    os.makedirs(dest_dir, exist_ok=True)

                    if names[k].rpartition(".")[0]:
                        dest_name = names[k].rpartition(".")[0] + ".mat"
                    else:
                        dest_name = names[k] + ".mat"
                    dest_name = os.path.join(dest_dir, dest_name)
                    print("Shape:", offset.shape, "Saving to", dest_name)

                    data_dict = {"mat": offset}

                    scipy.io.savemat(dest_name, data_dict, do_compression=True)
                    try:
                        scipy.io.loadmat(dest_name)
                    except Exception as e:
                        print(e)
                        scipy.io.savemat(dest_name, data_dict, do_compression=False)

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        Log.info("Test Time {batch_time.sum:.3f}s".format(batch_time=self.batch_time))

    def extract_offset(self, inputs):
        if self.sscrop:
            outputs = self.sscrop_test(inputs, self.crop_size)
        elif self.configer.get("test", "mode") == "ss_test":
            outputs = self.ss_test(inputs)

        offsets = []
        logits = []

        for mask_logits, dir_logits, img in zip(*outputs[:2], inputs[0]):
            h, w = img.shape[1:]

            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            )
            dir_logits = F.interpolate(
                dir_logits.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            )

            logit = torch.softmax(dir_logits, dim=1)
            zero_mask = mask_logits.argmax(dim=1, keepdim=True) == 0
            logits.append(mask_logits[:, 1])

            offset = self._get_offset(mask_logits, dir_logits)
            offsets.append(offset)
        print([x.shape for x in offsets])
        return offsets, logits

    def _get_offset(self, mask_logits, dir_logits):

        edge_mask = mask_logits[:, 1] > 0.5
        dir_logits = torch.softmax(dir_logits, dim=1)
        n, _, h, w = dir_logits.shape

        keep_mask = edge_mask

        dir_label = torch.argmax(dir_logits, dim=1).float()
        offset = DTOffsetHelper.label_to_vector(dir_label)
        offset = offset.permute(0, 2, 3, 1)
        offset[~keep_mask, :] = 0
        return offset

    def _flip(self, x, dim=-1):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(
            x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
        )
        return x[tuple(indices)]

    def _flip_offset(self, x):
        x = self._flip(x, dim=-1)
        if len(x.shape) == 4:
            return x[:, DTOffsetHelper.flipping_indices()]
        else:
            return x[DTOffsetHelper.flipping_indices()]

    def _flip_inputs(self, inputs):

        if self.size_mode == "fix_size":
            return [self._flip(x, -1) for x in inputs]
        else:
            return [[self._flip(x, -1) for x in xs] for xs in inputs]

    def _flip_outputs(self, outputs):
        funcs = [self._flip, self._flip_offset]
        if self.size_mode == "fix_size":
            return [f(x) for f, x in zip(funcs, outputs)]
        else:
            return [[f(x) for x in xs] for f, xs in zip(funcs, outputs)]

    def _tuple_sum(self, tup1, tup2, tup2_weight=1):
        """
        tup1 / tup2: tuple of tensors or tuple of list of tensors
        """

        if tup1 is None:
            if self.size_mode == "fix_size":
                return [y * tup2_weight for y in tup2]
            else:
                return [[y * tup2_weight for y in ys] for ys in tup2]
        else:
            if self.size_mode == "fix_size":
                return [x + y * tup2_weight for x, y in zip(tup1, tup2)]
            else:
                return [
                    [x + y * tup2_weight for x, y in zip(xs, ys)]
                    for xs, ys in zip(tup1, tup2)
                ]

    def _scale_ss_inputs(self, inputs, scale):
        n, c, h, w = inputs[0].shape
        size = (int(h * scale), int(w * scale))
        return [
            F.interpolate(inputs[0], size=size, mode="bilinear", align_corners=True),
        ], (h, w)

    def sscrop_test(self, inputs, crop_size, scale=1):
        """
        Currently, sscrop_test does not support diverse_size testing
        """
        scaled_inputs = inputs
        img = scaled_inputs[0]
        n, c, h, w = img.size(0), img.size(1), img.size(2), img.size(3)
        ori_h, ori_w = h, w
        full_probs = [torch.cuda.FloatTensor(n, dim, h, w).fill_(0) for dim in (2, 8)]
        count_predictions = [
            torch.cuda.FloatTensor(n, dim, h, w).fill_(0) for dim in (2, 8)
        ]

        crop_counter = 0

        height_starts = self._decide_intersection(h, crop_size[0])
        width_starts = self._decide_intersection(w, crop_size[1])

        for height in height_starts:
            for width in width_starts:
                crop_inputs = [
                    x[..., height : height + crop_size[0], width : width + crop_size[1]]
                    for x in scaled_inputs
                ]
                prediction = self.ss_test(crop_inputs)

                for j in range(2):
                    count_predictions[j][
                        :,
                        :,
                        height : height + crop_size[0],
                        width : width + crop_size[1],
                    ] += 1
                    full_probs[j][
                        :,
                        :,
                        height : height + crop_size[0],
                        width : width + crop_size[1],
                    ] += prediction[j]
                crop_counter += 1
                Log.info("predicting {:d}-th crop".format(crop_counter))

        for j in range(2):
            full_probs[j] /= count_predictions[j]
            full_probs[j] = F.interpolate(
                full_probs[j], size=(ori_h, ori_w), mode="bilinear", align_corners=True
            )
        return full_probs

    def _scale_ss_outputs(self, outputs, size):
        return [
            F.interpolate(x, size=size, mode="bilinear", align_corners=True)
            for x in outputs
        ]

    def ss_test(self, inputs, scale=1):
        if self.size_mode == "fix_size":

            scaled_inputs, orig_size = self._scale_ss_inputs(inputs, scale)
            print([x.shape for x in scaled_inputs])

            start = timeit.default_timer()
            outputs = list(self.seg_net.forward(*scaled_inputs))
            if len(outputs) == 3:
                outputs = (outputs[0], outputs[2])
            else:
                outputs[0] = F.softmax(outputs[0], dim=1)
            torch.cuda.synchronize()
            end = timeit.default_timer()

            return self._scale_ss_outputs(outputs, orig_size)

        else:
            device_ids = self.configer.get("gpu")
            replicas = nn.parallel.replicate(self.seg_net.module, device_ids)
            scaled_inputs, ori_sizes, outputs = [], [], []

            for *i, d in zip(*inputs, device_ids):
                scaled_i, ori_size_i = self._scale_ss_inputs(
                    [x.unsqueeze(0) for x in i], scale
                )
                scaled_inputs.append([x.cuda(d, non_blocking=True) for x in scaled_i])
                ori_sizes.append(ori_size_i)

            scaled_outputs = nn.parallel.parallel_apply(
                replicas[: len(scaled_inputs)], scaled_inputs
            )

            for o, ori_size in zip(scaled_outputs, ori_sizes):
                o = self._scale_ss_outputs(o, ori_size)
                if len(o) == 3:
                    o = (o[0], o[2])
                outputs.append([x.squeeze(0) for x in o])
            outputs = list(map(list, zip(*outputs)))
            return outputs

    def _decide_intersection(self, total_length, crop_length, crop_stride_ratio=1 / 3):
        stride = int(crop_length * crop_stride_ratio)  # set the stride as the paper do
        times = (total_length - crop_length) // stride + 1
        cropped_starting = []
        for i in range(times):
            cropped_starting.append(stride * i)

        if total_length - cropped_starting[-1] > crop_length:
            cropped_starting.append(
                total_length - crop_length
            )  # must cover the total image

        return cropped_starting


if __name__ == "__main__":
    pass
