##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret, LayneH
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
import time
import timeit
import pdb
import cv2
import scipy
import collections

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as dcrf_utils

from lib.utils.helpers.file_helper import FileHelper
from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.metrics.running_score import RunningScore
from lib.vis.seg_visualizer import SegVisualizer
from lib.vis.palette import get_cityscapes_colors, get_ade_colors, get_lip_colors
from lib.vis.palette import get_pascal_context_colors, get_cocostuff_colors, get_pascal_voc_colors
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from scipy import ndimage
from PIL import Image
from math import ceil


class Tester(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.seg_data_loader = DataLoader(configer)
        self.save_dir = self.configer.get('test', 'out_dir')
        self.seg_net = None
        self.test_loader = None
        self.test_size = None
        self.infer_time = 0
        self.infer_cnt = 0
        self._init_model()

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        if 'test' in self.save_dir:
            self.test_loader = self.seg_data_loader.get_testloader()
            self.test_size = len(self.test_loader) * self.configer.get('test', 'batch_size')
        else:
            self.test_loader = self.seg_data_loader.get_valloader()
            self.test_size = len(self.test_loader) * self.configer.get('val', 'batch_size')

        self.seg_net.eval()

    def __relabel(self, label_map):
        height, width = label_map.shape
        label_dst = np.zeros((height, width), dtype=np.uint8)
        for i in range(self.configer.get('data', 'num_classes')):
            label_dst[label_map == i] = self.configer.get('data', 'label_list')[i]

        label_dst = np.array(label_dst, dtype=np.uint8)

        return label_dst

    def test(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()
        image_id = 0

        Log.info('save dir {}'.format(self.save_dir))
        FileHelper.make_dirs(self.save_dir, is_file=False)

        if self.configer.get('dataset') in ['cityscapes', 'gta5']:
            colors = get_cityscapes_colors()
        elif self.configer.get('dataset') == 'ade20k':
            colors = get_ade_colors()
        elif self.configer.get('dataset') == 'lip':
            colors = get_lip_colors()
        elif self.configer.get('dataset') == 'pascal_context':
            colors = get_pascal_context_colors()
        elif self.configer.get('dataset') == 'pascal_voc':
            colors = get_pascal_voc_colors()
        elif self.configer.get('dataset') == 'coco_stuff':
            colors = get_cocostuff_colors()
        else:
            raise RuntimeError("Unsupport colors")

        save_prob = False
        if self.configer.get('test', 'save_prob'):
            save_prob = self.configer.get('test', 'save_prob')
            def softmax(X, axis=0):
                max_prob = np.max(X, axis=axis, keepdims=True)
                X -= max_prob
                X = np.exp(X)
                sum_prob = np.sum(X, axis=axis, keepdims=True)
                X /= sum_prob
                return X

        # for j, data_dict in reversed(list(enumerate(self.test_loader))):
        for j, data_dict in enumerate(self.test_loader):
            inputs = data_dict['img']
            names = data_dict['name']
            metas = data_dict['meta']
            
            if 'val' in self.save_dir and os.environ.get('save_gt_label'):
                labels = data_dict['labelmap']

            with torch.no_grad():
                # Forward pass.
                if self.configer.exists('data', 'use_offset') and self.configer.get('data', 'use_offset') == 'offline':
                    offset_h_maps = data_dict['offsetmap_h']
                    offset_w_maps = data_dict['offsetmap_w']
                    outputs = self.offset_test(inputs, offset_h_maps, offset_w_maps) 
                elif self.configer.get('test', 'mode') == 'ss_test':
                    outputs = self.ss_test(inputs)
                elif self.configer.get('test', 'mode') == 'ms_test':
                    outputs = self.ms_test(inputs)
                elif self.configer.get('test', 'mode') == 'ms_test_depth':
                    outputs = self.ms_test_depth(inputs, names)
                elif self.configer.get('test', 'mode') == 'sscrop_test':
                    crop_size = self.configer.get('test', 'crop_size')
                    outputs = self.sscrop_test(inputs, crop_size)
                elif self.configer.get('test', 'mode') == 'mscrop_test':
                    crop_size = self.configer.get('test', 'crop_size')
                    outputs = self.mscrop_test(inputs, crop_size)
                elif self.configer.get('test', 'mode') == 'crf_ss_test':
                    outputs = self.ss_test(inputs)
                    outputs = self.dense_crf_process(inputs, outputs)

                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()
                    n = outputs.shape[0]
                else:
                    outputs = [output.permute(0, 2, 3, 1).cpu().numpy().squeeze() for output in outputs]
                    n = len(outputs)

                for k in range(n):
                    image_id += 1
                    ori_img_size = metas[k]['ori_img_size']
                    border_size = metas[k]['border_size']
                    logits = cv2.resize(outputs[k][:border_size[1], :border_size[0]],
                                        tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)

                    # save the logits map
                    if self.configer.get('test', 'save_prob'):
                        prob_path = os.path.join(self.save_dir, "prob/", '{}.npy'.format(names[k]))
                        FileHelper.make_dirs(prob_path, is_file=True)
                        np.save(prob_path, softmax(logits, axis=-1))

                    label_img = np.asarray(np.argmax(logits, axis=-1), dtype=np.uint8)
                    if self.configer.exists('data', 'reduce_zero_label') and self.configer.get('data', 'reduce_zero_label'):
                        label_img = label_img + 1
                        label_img = label_img.astype(np.uint8)
                    if self.configer.exists('data', 'label_list'):
                        label_img_ = self.__relabel(label_img)
                    else:
                        label_img_ = label_img
                    label_img_ = Image.fromarray(label_img_, 'P')
                    Log.info('{:4d}/{:4d} label map generated'.format(image_id, self.test_size))
                    label_path = os.path.join(self.save_dir, "label/", '{}.png'.format(names[k]))
                    FileHelper.make_dirs(label_path, is_file=True)
                    ImageHelper.save(label_img_, label_path)

                    # colorize the label-map
                    if os.environ.get('save_gt_label'):
                        # label_img = np.asarray(labels[k], dtype=np.uint8)
                        if self.configer.exists('data', 'reduce_zero_label') and self.configer.get('data', 'reduce_zero_label'):
                            label_img = labels[k] + 1
                            label_img = np.asarray(label_img, dtype=np.uint8)
                        color_img_ = Image.fromarray(label_img)
                        color_img_.putpalette(colors)
                        vis_path = os.path.join(self.save_dir, "gt_vis/", '{}.png'.format(names[k]))
                        FileHelper.make_dirs(vis_path, is_file=True)
                        ImageHelper.save(color_img_, save_path=vis_path)
                    else:
                        color_img_ = Image.fromarray(label_img)
                        color_img_.putpalette(colors)
                        vis_path = os.path.join(self.save_dir, "vis/", '{}.png'.format(names[k]))
                        FileHelper.make_dirs(vis_path, is_file=True)
                        ImageHelper.save(color_img_, save_path=vis_path)

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        # Print the log info & reset the states.
        Log.info('Test Time {batch_time.sum:.3f}s'.format(batch_time=self.batch_time))


    def offset_test(self, inputs, offset_h_maps, offset_w_maps, scale=1):
        if isinstance(inputs, torch.Tensor):
            n, c, h, w = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
            start = timeit.default_timer()
            outputs = self.seg_net.forward(inputs, offset_h_maps, offset_w_maps)
            torch.cuda.synchronize()
            end = timeit.default_timer()

            if (self.configer.get('loss', 'loss_type') == "fs_auxce_loss") or (self.configer.get('loss', 'loss_type') == "triple_auxce_loss"):
                outputs = outputs[-1]
            elif self.configer.get('loss', 'loss_type') == "pyramid_auxce_loss":
                outputs = outputs[1] + outputs[2] + outputs[3] + outputs[4]

            outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
            return outputs
        else:
            raise RuntimeError("Unsupport data type: {}".format(type(inputs)))


    def ss_test(self, inputs, scale=1):
        if isinstance(inputs, torch.Tensor):
            n, c, h, w = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
            scaled_inputs = F.interpolate(inputs, size=(int(h*scale), int(w*scale)), mode="bilinear", align_corners=True)
            start = timeit.default_timer()
            outputs = self.seg_net.forward(scaled_inputs)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            outputs = outputs[-1]
            outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
            return outputs
        elif isinstance(inputs, collections.Sequence):
            device_ids = self.configer.get('gpu')
            replicas = nn.parallel.replicate(self.seg_net.module, device_ids)
            scaled_inputs, ori_size, outputs = [], [], []
            for i, d in zip(inputs, device_ids):
                h, w = i.size(1), i.size(2)
                ori_size.append((h, w))
                i = F.interpolate(i.unsqueeze(0), size=(int(h*scale), int(w*scale)), mode="bilinear", align_corners=True)
                scaled_inputs.append(i.cuda(d, non_blocking=True))
            scaled_outputs = nn.parallel.parallel_apply(replicas[:len(scaled_inputs)], scaled_inputs)
            for i, output in enumerate(scaled_outputs):
                outputs.append(F.interpolate(output[-1], size=ori_size[i], mode='bilinear', align_corners=True))
            return outputs
        else:
            raise RuntimeError("Unsupport data type: {}".format(type(inputs)))


    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]


    def sscrop_test(self, inputs, crop_size, scale=1):
        '''
        Currently, sscrop_test does not support diverse_size testing
        '''
        n, c, ori_h, ori_w = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
        scaled_inputs = F.interpolate(inputs, size=(int(ori_h*scale), int(ori_w*scale)), mode="bilinear", align_corners=True)
        n, c, h, w = scaled_inputs.size(0), scaled_inputs.size(1), scaled_inputs.size(2), scaled_inputs.size(3)
        full_probs = torch.cuda.FloatTensor(n, self.configer.get('data', 'num_classes'), h, w).fill_(0)
        count_predictions = torch.cuda.FloatTensor(n, self.configer.get('data', 'num_classes'), h, w).fill_(0)

        crop_counter = 0

        height_starts = self._decide_intersection(h, crop_size[0])
        width_starts = self._decide_intersection(w, crop_size[1])

        for height in height_starts:
            for width in width_starts:
                crop_inputs = scaled_inputs[:, :, height:height+crop_size[0], width:width + crop_size[1]]
                prediction = self.ss_test(crop_inputs)
                count_predictions[:, :, height:height+crop_size[0], width:width + crop_size[1]] += 1
                full_probs[:, :, height:height+crop_size[0], width:width + crop_size[1]] += prediction 
                crop_counter += 1
                Log.info('predicting {:d}-th crop'.format(crop_counter))

        full_probs /= count_predictions
        full_probs = F.interpolate(full_probs, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
        return full_probs


    def ms_test(self, inputs):
        if isinstance(inputs, torch.Tensor):  
            n, c, h, w = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
            full_probs = torch.cuda.FloatTensor(n, self.configer.get('data', 'num_classes'), h, w).fill_(0)
            if self.configer.exists('test', 'scale_weights'):
                for scale, weight in zip(self.configer.get('test', 'scale_search'), self.configer.get('test', 'scale_weights')):
                    probs = self.ss_test(inputs, scale)
                    flip_probs = self.ss_test(self.flip(inputs, 3), scale)
                    probs = probs + self.flip(flip_probs, 3)
                    full_probs += weight * probs
                return full_probs
            else:
                for scale in self.configer.get('test', 'scale_search'):
                    probs = self.ss_test(inputs, scale)
                    flip_probs = self.ss_test(self.flip(inputs, 3), scale)
                    probs = probs + self.flip(flip_probs, 3)
                    full_probs += probs
                return full_probs

        elif isinstance(inputs, collections.Sequence):
            device_ids = self.configer.get('gpu')
            full_probs = [torch.zeros(1, self.configer.get('data', 'num_classes'), 
                i.size(1), i.size(2)).cuda(device_ids[index], non_blocking=True)
                for index, i in enumerate(inputs)]
            flip_inputs = [self.flip(i, 2) for i in inputs]

            if self.configer.exists('test', 'scale_weights'):
                for scale, weight in zip(self.configer.get('test', 'scale_search'), self.configer.get('test', 'scale_weights')):
                    probs = self.ss_test(inputs, scale)
                    flip_probs = self.ss_test(flip_inputs, scale)
                    for i in range(len(inputs)):
                        full_probs[i] += weight * (probs[i] + self.flip(flip_probs[i], 3))
                return full_probs
            else:
                for scale in self.configer.get('test', 'scale_search'):
                    probs = self.ss_test(inputs, scale)
                    flip_probs = self.ss_test(flip_inputs, scale)
                    for i in range(len(inputs)):
                        full_probs[i] += (probs[i] + self.flip(flip_probs[i], 3))
                return full_probs

        else:
            raise RuntimeError("Unsupport data type: {}".format(type(inputs)))


    def ms_test_depth(self, inputs, names):
        prob_list = []
        scale_list = []

        if isinstance(inputs, torch.Tensor):  
            n, c, h, w = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
            full_probs = torch.cuda.FloatTensor(n, self.configer.get('data', 'num_classes'), h, w).fill_(0)

            for scale in self.configer.get('test', 'scale_search'):
                probs = self.ss_test(inputs, scale)
                flip_probs = self.ss_test(self.flip(inputs, 3), scale)
                probs = probs + self.flip(flip_probs, 3)
                prob_list.append(probs)
                scale_list.append(scale)

            full_probs = self.fuse_with_depth(prob_list, scale_list, names)
            return full_probs

        else:
            raise RuntimeError("Unsupport data type: {}".format(type(inputs)))


    def fuse_with_depth(self, probs, scales, names):
        MAX_DEPTH = 63
        POWER_BASE = 0.8
        if 'test' in self.save_dir:
            stereo_path = "/msravcshare/dataset/cityscapes/stereo/test/"
        else:
            stereo_path = "/msravcshare/dataset/cityscapes/stereo/val/"

        n, c, h, w = probs[0].size(0), probs[0].size(1), probs[0].size(2), probs[0].size(3)
        full_probs = torch.cuda.FloatTensor(n, self.configer.get('data', 'num_classes'), h, w).fill_(0)

        for index, name in enumerate(names):
            stereo_map = cv2.imread(stereo_path + name + '.png', -1)
            depth_map = stereo_map / 256.0
            depth_map = 0.5 / depth_map
            depth_map = 500 * depth_map

            depth_map = np.clip(depth_map, 0, MAX_DEPTH)
            depth_map = depth_map // (MAX_DEPTH // len(scales))

            for prob, scale in zip(probs, scales):
                scale_index = self._locate_scale_index(scale, scales)
                weight_map = np.abs(depth_map - scale_index)
                weight_map = np.power(POWER_BASE, weight_map)
                weight_map = cv2.resize(weight_map, (w, h))
                full_probs[index, :, :, :] += torch.from_numpy(np.expand_dims(weight_map, axis=0)).type(torch.cuda.FloatTensor) * prob[index, :, :, :]

        return full_probs

    @staticmethod
    def _locate_scale_index(scale, scales):
        for idx, s in enumerate(scales):
            if scale == s:
                return idx
        return 0


    def ms_test_wo_flip(self, inputs):
        if isinstance(inputs, torch.Tensor):  
            n, c, h, w = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
            full_probs = torch.cuda.FloatTensor(n, self.configer.get('data', 'num_classes'), h, w).fill_(0)
            for scale in self.configer.get('test', 'scale_search'):
                probs = self.ss_test(inputs, scale)
                full_probs += probs
            return full_probs
        elif isinstance(inputs, collections.Sequence):
            device_ids = self.configer.get('gpu')
            full_probs = [torch.zeros(1, self.configer.get('data', 'num_classes'), 
                i.size(1), i.size(2)).cuda(device_ids[index], non_blocking=True)
                for index, i, in enumerate(inputs)]
            for scale in self.configer.get('test', 'scale_search'):
                probs = self.ss_test(inputs, scale)
                for i in range(len(inputs)):
                    full_probs[i] += probs[i]
            return full_probs
        else:
            raise RuntimeError("Unsupport data type: {}".format(type(inputs)))


    def mscrop_test(self, inputs, crop_size):  
        '''
        Currently, mscrop_test does not support diverse_size testing
        '''
        n, c, h, w = inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)
        full_probs = torch.cuda.FloatTensor(n, self.configer.get('data', 'num_classes'), h, w).fill_(0)
        for scale in self.configer.get('test', 'scale_search'):
            Log.info('Scale {0:.2f} prediction'.format(scale))
            if scale < 1:
                probs = self.ss_test(inputs, scale)
                flip_probs = self.ss_test(self.flip(inputs, 3), scale)
                probs = probs + self.flip(flip_probs, 3)
                full_probs += probs
            else:
                probs = self.sscrop_test(inputs, crop_size, scale)
                flip_probs = self.sscrop_test(self.flip(inputs, 3), crop_size, scale)
                probs = probs + self.flip(flip_probs, 3)
                full_probs += probs
        return full_probs


    def _decide_intersection(self, total_length, crop_length):
        stride = crop_length
        times = (total_length - crop_length) // stride + 1
        cropped_starting = []
        for i in range(times):
            cropped_starting.append(stride*i)
        if total_length - cropped_starting[-1] > crop_length:
            cropped_starting.append(total_length - crop_length)  # must cover the total image
        return cropped_starting


    def dense_crf_process(self, images, outputs):
        '''
        Reference: https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/utils/crf.py
        '''
        # hyperparameters of the dense crf 
        # baseline = 79.5
        # bi_xy_std = 67, 79.1
        # bi_xy_std = 20, 79.6
        # bi_xy_std = 10, 79.7
        # bi_xy_std = 10, iter_max = 20, v4 79.7
        # bi_xy_std = 10, iter_max = 5, v5 79.7
        # bi_xy_std = 5, v3 79.7
        iter_max = 10
        pos_w = 3
        pos_xy_std = 1
        bi_w = 4
        bi_xy_std = 10
        bi_rgb_std = 3

        b = images.size(0)
        mean_vector = np.expand_dims(np.expand_dims(np.transpose(np.array([102.9801, 115.9465, 122.7717])), axis=1), axis=2)
        outputs = F.softmax(outputs, dim=1)
        for i in range(b):
            unary = outputs[i].data.cpu().numpy()
            C, H, W = unary.shape
            unary = dcrf_utils.unary_from_softmax(unary)
            unary = np.ascontiguousarray(unary)
            
            image = np.ascontiguousarray(images[i]) + mean_vector
            image = image.astype(np.ubyte)
            image = np.ascontiguousarray(image.transpose(1, 2, 0))

            d = dcrf.DenseCRF2D(W, H, C)
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w)
            d.addPairwiseBilateral(sxy=bi_xy_std, srgb=bi_rgb_std, rgbim=image, compat=bi_w)
            out_crf = np.array(d.inference(iter_max))
            outputs[i] = torch.from_numpy(out_crf).cuda().view(C, H, W)

        return outputs


if __name__ == "__main__":
    pass
