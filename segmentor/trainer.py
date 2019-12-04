##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret, JingyiXie, LangHuang
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

import time

import os
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.metrics.F1_running_score import F1RunningScore
from lib.metrics.running_score import RunningScore, MultiLabelRunningScore
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from lib.utils.distributed import get_world_size, get_rank, is_distributed


class Trainer(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        
        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None

        self._init_model()
        self._init_running_score()

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)

        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.pixel_loss = self.loss_manager.get_seg_loss()
        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)        

    def _init_running_score(self):
        from lib.utils.helpers.offset_helper import DTOffsetConfig
        self.main_running_score_index = None
        if 'gum_one_stage_shift' in self.configer.get('loss', 'loss_type'):
            self.main_running_score_index = -1       
            self.running_score = [
                RunningScore(self.configer, num_classes=2, ignore_index=-1),
                RunningScore(self.configer)
            ]
        elif 'one_stage_shift' in self.configer.get('loss', 'loss_type'):
            self.main_running_score_index = -1       
            self.running_score = [
                RunningScore(self.configer, num_classes=2, ignore_index=-1),
                RunningScore(self.configer, num_classes=DTOffsetConfig.num_classes, ignore_index=-1),
                RunningScore(self.configer, num_classes=DTOffsetConfig.num_classes + 1, ignore_index=-1),
                RunningScore(self.configer)
            ]
        elif 'one_stage_relax_shift' in self.configer.get('loss', 'loss_type'):
            self.main_running_score_index = -1       
            self.running_score = [
                RunningScore(self.configer, num_classes=2, ignore_index=-1),
                MultiLabelRunningScore(),
                RunningScore(self.configer)
            ]
        elif 'one_stage_edge' in self.configer.get('loss', 'loss_type'):
            self.main_running_score_index = -1
            self.running_score = [
                RunningScore(self.configer, num_classes=2, ignore_index=-1),
                RunningScore(self.configer)
            ]
        elif 'angle_mse' in self.configer.get('loss', 'loss_type'):
            self.running_score = [
                RunningScore(self.configer, num_classes=2, ignore_index=-1),
                RunningScore(self.configer, num_classes=DTOffsetConfig.num_classes, ignore_index=-1)
            ]
        elif 'offset' in self.configer.get('loss', 'loss_type'):
            self.main_running_score_index = -1
            if self.configer.exists('data', 'pred_ml_dt_offset'):
                self.running_score = [
                    RunningScore(self.configer, num_classes=2, ignore_index=-1), 
                    MultiLabelRunningScore(),
                    MultiLabelRunningScore(),
                ]
            elif 'sobel_mask' in self.configer.get('loss', 'loss_type'):
                self.running_score = [
                    RunningScore(self.configer, num_classes=2, ignore_index=-1), 
                    RunningScore(self.configer, num_classes=DTOffsetConfig.num_classes, ignore_index=-1),
                    RunningScore(self.configer, num_classes=DTOffsetConfig.num_classes + 1, ignore_index=-1),
                ]
            elif 'mask' in self.configer.get('loss', 'loss_type'):
                self.running_score = [
                    RunningScore(self.configer, num_classes=2), 
                    RunningScore(self.configer, num_classes=DTOffsetConfig.num_classes, ignore_index=-1),
                    RunningScore(self.configer, num_classes=DTOffsetConfig.num_classes + 1, ignore_index=-1),
                ]
            elif os.environ.get('val_all_class'):
                # evaluate the h, w direction accuracy for each category
                self.running_score = [
                    RunningScore(self.configer, ignore_index=0, num_classes=3) 
                    for _ in range((self.configer.get('data', 'num_classes') + 1) * 2)
                ]
            else:
                # evaluate the h, w direction accuracy for all category
                self.running_score = [
                    RunningScore(self.configer, ignore_index=0, num_classes=3) 
                    for _ in range(2)
                ]
        else:
            self.running_score = RunningScore(self.configer)


    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                nbb_lr.append(value)
            else:
                bb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        self.pixel_loss.train()
        start_time = time.time()

        if "swa" in self.configer.get('lr', 'lr_policy'):
            normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get('solver', 'max_iters') - normal_max_iters) // 5 + 1

        if is_distributed():
            device = torch.device('cuda:{}'.format(get_rank()))
            self.train_loader.sampler.set_epoch(self.configer.get('epoch'))

        for i, data_dict in enumerate(self.train_loader):
            if self.configer.get('lr', 'metric') == 'iters':
                self.scheduler.step(self.configer.get('iters'))
            else:
                self.scheduler.step(self.configer.get('epoch'))

            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(self.configer.get('iters'),
                                           self.scheduler, self.optimizer, backbone_list=[0,])
            inputs = data_dict['img']

            if is_distributed():
                inputs = inputs.to(device)

            if self.configer.exists('data', 'use_edge') and self.configer.get('data', 'use_edge') == 'gscnn':
                targets = [
                    data_dict['labelmap'], 
                    data_dict['edgemap'],
                ]
            elif self.configer.exists('data', 'use_edge') and self.configer.get('data', 'use_edge') == 'ce2p':
                targets = [
                    data_dict['labelmap'], 
                    data_dict['maskmap'],
                ]
            elif self.configer.exists('data', 'pred_sw_offset') and 'two_stage' not in self.configer.get('network', 'model_name'):
                targets = [
                    data_dict['labelmap'], 
                    data_dict['offsetmap_h'], 
                    data_dict['offsetmap_w'],
                ]
            elif self.configer.exists('data', 'pred_dt_offset') and 'two_stage' not in self.configer.get('network', 'model_name'):
                targets = [
                    data_dict['labelmap'], 
                    data_dict['distance_map'], 
                    data_dict['angle_map'],
                ]  
            elif self.configer.exists('data', 'pred_ml_dt_offset'):
                targets = [
                    data_dict['labelmap'],
                    data_dict['distance_map'],
                    data_dict['multi_label_direction_map'],
                ]
            else:
                targets = data_dict['labelmap']
                if is_distributed():
                    targets = targets.to(device)

            self.data_time.update(time.time() - start_time)

            foward_start_time = time.time()

            if self.configer.exists('data', 'use_sw_offset'):
                outputs = self.seg_net(inputs, data_dict['offsetmap_h'], data_dict['offsetmap_w'])
            elif self.configer.exists('data', 'use_dt_offset'):
                outputs = self.seg_net(inputs, data_dict['distance_map'], data_dict['angle_map'])
            else:
                if self.configer.get('use_ground_truth'):
                    if self.configer.exists('data', 'use_edge') and self.configer.get('data', 'use_edge') == 'ce2p':
                        outputs = self.seg_net(inputs, targets[0])
                    else:
                        if "condition_mask_offset" in self.configer.get('network', 'model_name'):
                            outputs = self.seg_net(inputs, targets[0])
                        else:
                            outputs = self.seg_net(inputs, targets)
                else:
                    # Log.info('inputs shape: {}'.format(inputs.shape))
                    outputs = self.seg_net(inputs)
            self.foward_time.update(time.time() - foward_start_time)

            loss_start_time = time.time()
            if is_distributed():
                loss = self.pixel_loss(outputs, targets)
                loss = self.module_runner.to_device(loss)
                dist.reduce(loss, dst=0)
                loss = loss / get_world_size()
            else:
                loss = self.pixel_loss(outputs, targets, gathered=self.configer.get('network', 'gathered'))
            
            if self.configer.exists('train', 'loader') and self.configer.get('train', 'loader') == 'ade20k':
                batch_size = self.configer.get('train', 'batch_size')*self.configer.get('train', 'batch_per_gpu')
                self.train_losses.update(loss.item(), batch_size)
            else:
                self.train_losses.update(loss.item(), inputs.size(0))
            self.loss_time.update(time.time() - loss_start_time)

            backward_start_time = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.backward_time.update(time.time() - backward_start_time)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 and \
                (not is_distributed() or get_rank() == 0):
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                         'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                         'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                         self.configer.get('epoch'), self.configer.get('iters'),
                         self.configer.get('solver', 'display_iter'),
                         self.module_runner.get_lr(self.optimizer), batch_time=self.batch_time,
                         foward_time=self.foward_time, backward_time=self.backward_time, loss_time=self.loss_time,
                         data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.loss_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # save checkpoints for swa
            if 'swa' in self.configer.get('lr', 'lr_policy') and \
               self.configer.get('iters') > normal_max_iters and \
               ((self.configer.get('iters') - normal_max_iters) % swa_step_max_iters == 0 or \
                self.configer.get('iters') == self.configer.get('solver', 'max_iters')):
               self.optimizer.update_swa()

            if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            # if self.configer.get('epoch') % self.configer.get('solver', 'test_interval') == 0:
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

        self.configer.plus_one('epoch')


    def __val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()
        # device_ids = self.configer.get('gpu')
        device_ids = list(range(len(self.configer.get('gpu'))))
        size_mode = self.configer.get('val', 'data_transformer')['size_mode']
        if size_mode == "diverse_size":
            cudnn.benchmark = False
            assert self.configer.get('val', 'batch_size') <= len(device_ids)
            replicas = nn.parallel.replicate(self.seg_net.module, device_ids)

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:
                Log.info('{} images processed\n'.format(j))

            inputs = data_dict['img']
            # targets = data_dict['labelmap']
            if self.configer.exists('data', 'pred_sw_offset') and 'two_stage' not in self.configer.get('network', 'model_name'):
                targets = [
                    data_dict['labelmap'], 
                    data_dict['offsetmap_h'], 
                    data_dict['offsetmap_w'],
                ]
            elif self.configer.exists('data', 'pred_dt_offset') and 'two_stage' not in self.configer.get('network', 'model_name'):
                targets = [
                    data_dict['labelmap'], 
                    data_dict['distance_map'], 
                    data_dict['angle_map'],
                ]
            elif self.configer.exists('data', 'pred_ml_dt_offset'):
                targets = [
                    data_dict['labelmap'],
                    data_dict['distance_map'],
                    data_dict['multi_label_direction_map'],
                ]
            else:
                targets = data_dict['labelmap']

            if self.configer.get('dataset') == 'lip':
                inputs_rev = torch.flip(inputs, [3])
                targets_rev = torch.flip(targets, [2])

            with torch.no_grad():
                if self.configer.get('dataset') == 'lip':
                    inputs, inputs_rev, targets,targets_rev = self.module_runner.to_device(inputs, inputs_rev, targets, targets_rev)
                    inputs = torch.cat([inputs, inputs_rev], dim=0)
                    
                    if self.configer.get('use_ground_truth'):
                        targets = torch.cat([targets, targets_rev], dim=0)
                        outputs = self.seg_net(inputs, targets)
                    else:
                        outputs = self.seg_net(inputs)

                    outputs_ = self.module_runner.gather(outputs)
                    if isinstance(outputs_, (list, tuple)):
                        outputs_ = outputs_[-1]
                    outputs = outputs_[0:int(outputs_.size(0)/2),:,:,:].clone()
                    outputs_rev = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),:,:,:].clone()
                    if outputs_rev.shape[1] == 20:
                        outputs_rev[:,14,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),15,:,:]
                        outputs_rev[:,15,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),14,:,:]
                        outputs_rev[:,16,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),17,:,:]
                        outputs_rev[:,17,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),16,:,:]
                        outputs_rev[:,18,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),19,:,:]
                        outputs_rev[:,19,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),18,:,:]
                    outputs_rev = torch.flip(outputs_rev, [3])
                    outputs = (outputs + outputs_rev) / 2.
                    self._update_running_score(outputs, data_dict['meta'])

                elif size_mode == "diverse_size":
                    inputs_cuda, targets_cuda = [], []

                    if self.configer.get('use_ground_truth'):
                        for i, t, d in zip(inputs, targets, device_ids):
                            inputs_cuda.append([i.unsqueeze(0).cuda(d, non_blocking=True),
                                                t.unsqueeze(0).cuda(d, non_blocking=True)])
                            targets_cuda.append(t.unsqueeze(0).cuda(d))
                        outputs = nn.parallel.parallel_apply(replicas[:len(inputs)], inputs_cuda)
                    else:
                        if self.configer.exists('data', 'use_dt_offset'):
                            inputs = [inputs, data_dict['distance_map'], data_dict['angle_map']]
                            targets = [data_dict['labelmap']]
                        else:
                            inputs = [inputs]
                            targets = [targets]

                        def split_and_cuda(lst, device_ids):
                            assert isinstance(lst, list)

                            results = []
                            for *items, d in zip(*lst, device_ids):
                                if len(items) == 1:
                                    results.append(items[0].unsqueeze(0).cuda(d))
                                else:
                                    results.append([item.unsqueeze(0).cuda(d) for item in items])

                            return results

                        inputs_cuda = split_and_cuda(inputs, device_ids)
                        targets_cuda = split_and_cuda(targets, device_ids)
                        outputs = nn.parallel.parallel_apply(replicas[:len(inputs_cuda)], inputs_cuda)

                    for i in range(len(outputs)):
                        loss = self.pixel_loss(outputs[i], targets_cuda[i])
                        self.val_losses.update(loss.item(), 1)
                        if 'aux' in self.configer.get('loss', 'loss_type'):
                            self._update_running_score(outputs[i][-1], data_dict['meta'][i:i+1])
                        else:
                            self._update_running_score(outputs[i], data_dict['meta'][i:i+1])
                            
                else:
                    if not isinstance(targets, list):
                        targets = [targets]
                    inputs, *targets = self.module_runner.to_device(inputs, *targets)
                    if len(targets) == 1:
                        targets = targets[0]

                    if self.configer.exists('data', 'use_sw_offset'):
                        args = data_dict['offsetmap_h'], data_dict['offsetmap_w']
                        outputs = self.seg_net(inputs, *args)
                    elif self.configer.exists('data', 'use_dt_offset'):
                        args = data_dict['distance_map'], data_dict['angle_map']
                        outputs = self.seg_net(inputs, *args)
                    else:
                        if self.configer.get('use_ground_truth'):
                            if "condition_mask_offset" in self.configer.get('network', 'model_name'):
                                outputs = self.seg_net(inputs, targets[0])
                            else:
                                outputs = self.seg_net(inputs, targets)
                        else:
                            outputs = self.seg_net(inputs)   

                    # compute the losses
                    if self.configer.exists('data', 'use_edge') and self.configer.get('data', 'use_edge') == 'gscnn':
                        loss = self.pixel_loss(outputs, [data_dict['labelmap'], data_dict['edgemap']], gathered=self.configer.get('network', 'gathered'))
                    elif self.configer.exists('data', 'use_edge') \
                        or self.configer.exists('data', 'use_dt_offset') \
                        or self.configer.exists('data', 'use_sw_offset'):
                        try:
                            loss = self.pixel_loss(outputs, targets, gathered=self.configer.get('network', 'gathered'))
                        except AssertionError as e:
                            print(len(outputs), len(targets))
                    else:
                        if is_distributed():
                            loss = self.pixel_loss(outputs, targets)
                        else:
                            loss = self.pixel_loss(outputs, targets, gathered=self.configer.get('network', 'gathered'))
                    if not is_distributed():
                        outputs = self.module_runner.gather(outputs) 
                        if self.configer.exists('data', 'use_edge') and self.configer.get('data', 'use_edge') == 'gscnn':
                            outputs = outputs[0] 
                    self.val_losses.update(loss.item(), inputs.size(0))

                    # compute the accuracy of the offset directions / angles
                    if 'gum_one_stage_shift' in self.configer.get('loss', 'loss_type'):
                        self._update_gum_one_stage_shift_score(outputs, data_dict['meta'])
                    elif 'one_stage_shift' in self.configer.get('loss', 'loss_type'):
                        self._update_one_stage_shift_score(outputs, data_dict['meta'])
                    elif 'one_stage_relax_shift' in self.configer.get('loss', 'loss_type'):
                        self._update_one_stage_relax_shift_score(outputs, data_dict['meta'])
                    elif 'one_stage_edge' in self.configer.get('loss', 'loss_type'):
                        self._update_one_stage_edge_score(outputs, data_dict['meta'])
                    elif 'angle' in self.configer.get('loss', 'loss_type'):
                        self._update_angle_score(outputs, data_dict['meta']) 
                    elif 'offset_loss' in self.configer.get('loss', 'loss_type'):
                        if 'multi_label' in self.configer.get('loss', 'loss_type') \
                            or 'relax' in self.configer.get('loss', 'loss_type'):
                            self._update_multi_label_dt_offset_score(outputs, data_dict['meta'])
                        elif 'sobel_mask' in self.configer.get('loss', 'loss_type'):
                            self._update_sobel_mask_dt_offset_score(outputs, data_dict['meta'])
                        elif 'side_mask_dt' in self.configer.get('loss', 'loss_type'):
                            outputs = [outputs[3], outputs[7]]
                            self._update_mask_dt_offset_score(outputs, data_dict['meta'])
                        elif 'mask_dt' in self.configer.get('loss', 'loss_type'):
                            self._update_mask_dt_offset_score(outputs, data_dict['meta'])
                        elif 'mask_sw' in self.configer.get('loss', 'loss_type'): 
                            self._update_mask_sw_offset_score(outputs, data_dict['meta'])
                        elif os.environ.get('val_seg'):
                            self._update_running_score(outputs[-1], data_dict['meta'])
                        else:
                            idx = 0
                            if os.environ.get('val_all_class'):
                                lst = [None] + list(range(self.configer.get('data', 'num_classes')))
                            else:
                                lst = [None]
                            for class_id in lst:
                                for direction in 'hw':
                                    self._update_offset_score(outputs[0], outputs[1], data_dict['meta'], class_id=class_id, direction=direction, idx=idx)
                                    idx += 1
                    elif isinstance(outputs, tuple) or isinstance(outputs, list):
                        self._update_running_score(outputs[-1], data_dict['meta'])
                    else:
                        self._update_running_score(outputs, data_dict['meta'])

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        if not isinstance(self.running_score, list):
            self.configer.update(
                ['performance'], 
                self.running_score.get_mean_iou()
            )

        if self.main_running_score_index is not None:
            self.configer.update(
                ['performance'],
                self.running_score[self.main_running_score_index].get_pixel_acc()
            )
        
        self.configer.update(['val_loss'], self.val_losses.avg)
        self.module_runner.save_net(self.seg_net, save_mode='performance')
        self.module_runner.save_net(self.seg_net, save_mode='val_loss')
        cudnn.benchmark = True

        # Print the log info & reset the states.
        if not is_distributed() or get_rank() == 0:
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            if isinstance(self.running_score, list):
                for idx, rs in enumerate(self.running_score):
                    Log.info('Result for {}'.format(idx))
                    Log.info('Mean IOU: {}\n'.format(rs.get_mean_iou()))
                    Log.info('Pixel ACC: {}\n'.format(rs.get_pixel_acc()))
            else:    
                Log.info('Mean IOU: {}\n'.format(self.running_score.get_mean_iou()))
                Log.info('Pixel ACC: {}\n'.format(self.running_score.get_pixel_acc()))
            
        self.batch_time.reset()
        self.val_losses.reset()
        if isinstance(self.running_score, list):
            for running_score in self.running_score:
                running_score.reset()
        else:
            self.running_score.reset()
        self.seg_net.train()
        self.pixel_loss.train()

    def _update_running_score(self, pred, metas):
        pred = pred.permute(0, 2, 3, 1)
        for i in range(pred.size(0)):
            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            ori_target = metas[i]['ori_target']
            total_logits = cv2.resize(pred[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            labelmap = np.argmax(total_logits, axis=-1)
            self.running_score.update(labelmap[None], ori_target[None])

    def _print_running_score(self, show_miou=True):
        if isinstance(self.running_score, list):
            for idx, rs in enumerate(self.running_score):
                Log.info('Result for {}'.format(idx))
                if show_miou:
                    Log.info('Mean IOU: {}\n'.format(rs.get_mean_iou()))
                Log.info('Pixel ACC: {}\n'.format(rs.get_pixel_acc()))
        else:    
            if show_miou:
                Log.info('Mean IOU: {}\n'.format(self.running_score.get_mean_iou()))
            Log.info('Pixel ACC: {}\n'.format(self.running_score.get_pixel_acc()))   

    def _update_gum_one_stage_shift_score(self, outputs, metas):

        from lib.utils.helpers.offset_helper import DTOffsetHelper, DTOffsetConfig
        mask_maps, _, seg_maps = outputs
        mask_maps, seg_maps = map(lambda x: x.permute(0, 2, 3, 1), (mask_maps, seg_maps))
    
        for i in range(mask_maps.size(0)):

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            distance_map = metas[i]['ori_distance_map']
            seg_label_map = metas[i]['ori_target']

            gt_mask_label = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map)

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)    
            seg_logits = cv2.resize(seg_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)  

            seg_pred = np.argmax(seg_logits, axis=-1)                                                                     
            mask_pred = np.argmax(mask_logits, axis=-1)

            self.running_score[0].update(mask_pred[None], gt_mask_label[None])
            self.running_score[1].update(seg_pred[None], seg_label_map[None])

    def _update_one_stage_shift_score(self, outputs, metas):

        from lib.utils.helpers.offset_helper import DTOffsetHelper, DTOffsetConfig
        mask_maps, dir_maps, _, seg_maps = outputs
        mask_maps, dir_maps, seg_maps = map(lambda x: x.permute(0, 2, 3, 1), (mask_maps, dir_maps, seg_maps))
    
        for i in range(mask_maps.size(0)):

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            distance_map = metas[i]['ori_distance_map']
            angle_map = metas[i]['ori_angle_map']
            seg_label_map = metas[i]['ori_target']

            gt_mask_label = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map)

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            dir_logits = cv2.resize(dir_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)      
            seg_logits = cv2.resize(seg_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)  

            seg_pred = np.argmax(seg_logits, axis=-1)                                                                     
            mask_pred = np.argmax(mask_logits, axis=-1)
            dir_pred = np.argmax(dir_logits, axis=-1)

            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map, 
                extra_ignore_mask=mask_pred == 0
            )
            self.running_score[0].update(mask_pred[None], gt_mask_label[None])

            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map, 
                extra_ignore_mask=mask_pred == 0
            ) 
            self.running_score[1].update(dir_pred[None], gt_dir_label[None])

            dir_pred[mask_pred == 0] = DTOffsetConfig.num_classes
            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map, 
                extra_ignore_mask=gt_mask_label == 0,
            )
            self.running_score[2].update(dir_pred[None], gt_dir_label[None]) 

            self.running_score[3].update(seg_pred[None], seg_label_map[None])

    def _update_one_stage_relax_shift_score(self, outputs, metas):

        from lib.utils.helpers.offset_helper import DTOffsetHelper, DTOffsetConfig
        mask_maps, dir_maps, _, seg_maps = outputs
        mask_maps, dir_maps, seg_maps = map(lambda x: x.permute(0, 2, 3, 1), (mask_maps, dir_maps, seg_maps))

        if 'relax' in self.configer.get('loss', 'loss_type'):
            dir_maps = torch.softmax(dir_maps, dim=-1)
        else:
            dir_maps = torch.sigmoid(dir_maps)

        for i in range(mask_maps.size(0)):

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            distance_map = metas[i]['ori_distance_map']
            seg_label_map = metas[i]['ori_target']
            dir_label_map = metas[i]['ori_multi_label_direction_map']

            gt_mask_label = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map)

            dir_label_map = DTOffsetHelper.encode_multi_labels(dir_label_map)
            dir_label_map[seg_label_map == -1, :] = -1

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            dir_logits = cv2.resize(dir_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)      
            seg_logits = cv2.resize(seg_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)  

            seg_pred = np.argmax(seg_logits, axis=-1)                                                                     
            mask_pred = np.argmax(mask_logits, axis=-1)

            dir_pred = self._get_multilabel_prediction(
                dir_logits, 
                no_offset_mask=mask_pred == 0,
                topk=8
            )

            self.running_score[0].update(mask_pred[None], gt_mask_label[None])
            self.running_score[1].update(dir_pred, dir_label_map, gt_mask_label == 1)
            self.running_score[2].update(seg_pred[None], seg_label_map[None])

    def _update_one_stage_edge_score(self, outputs, metas):

        from lib.utils.helpers.offset_helper import DTOffsetHelper
        mask_maps,  _, seg_maps = outputs
        mask_maps, seg_maps = map(lambda x: x.permute(0, 2, 3, 1), (mask_maps, seg_maps))
    
        for i in range(mask_maps.size(0)):

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            distance_map = metas[i]['ori_distance_map']
            angle_map = metas[i]['ori_angle_map']
            seg_label_map = metas[i]['ori_target']

            gt_mask_label = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map)

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC) 
            seg_logits = cv2.resize(seg_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)       
            seg_pred = np.argmax(seg_logits, axis=-1)                                                                     
            mask_pred = np.argmax(mask_logits, axis=-1)

            self.running_score[0].update(mask_pred[None], gt_mask_label[None])
            self.running_score[1].update(seg_pred[None], seg_label_map[None])
  
    def _update_angle_score(self, outputs, metas):
        from lib.utils.helpers.offset_helper import DTOffsetHelper
        mask_maps, dir_maps = map(lambda x: x.permute(0, 2, 3, 1), outputs)
    
        for i in range(mask_maps.size(0)):

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            distance_map = metas[i]['ori_distance_map']
            angle_map = metas[i]['ori_angle_map']
            seg_label_map = metas[i]['ori_target']

            gt_mask_label = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map)

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            dir_vectors = cv2.resize(dir_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_NEAREST)      
            mask_pred = np.argmax(mask_logits, axis=-1)
            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map, 
                distance_map=distance_map,
                extra_ignore_mask=mask_pred == 0
            )
            dir_pred = DTOffsetHelper.vector_to_label(dir_vectors)

            self.running_score[0].update(mask_pred[None], gt_mask_label[None])
            self.running_score[1].update(dir_pred[None], gt_dir_label[None])            

    def _update_mask_sw_offset_score(self, outputs, metas):
        mask_maps, dir_maps = map(lambda x: x.permute(0, 2, 3, 1), outputs)
        for i in range(mask_maps.size(0)):

            mask_label_map, dir_label_map = map(
                lambda x: x.cpu().numpy(),
                self.pixel_loss.module._convert_label(metas[i]['ori_offset_h'], metas[i]['ori_offset_w'])
            )

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            ori_target = metas[i]['ori_target']
            gt_offset_h = metas[i]['ori_offset_h']
            gt_offset_w = metas[i]['ori_offset_w']

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            dir_logits = cv2.resize(dir_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)                                      
            mask_pred = np.argmax(mask_logits, axis=-1)
            dir_pred = np.argmax(dir_logits, axis=-1)

            dir_label_map[mask_pred == 0] = -1

            self.running_score[0].update(mask_pred[None], mask_label_map[None])
            self.running_score[1].update(dir_pred[None], dir_label_map[None])

    def _update_mask_dt_offset_score(self, outputs, metas):
        from lib.utils.helpers.offset_helper import DTOffsetHelper, DTOffsetConfig       
        mask_maps, dir_maps = map(lambda x: x.permute(0, 2, 3, 1), outputs)

        for i in range(mask_maps.size(0)):

            distance_map = metas[i]['ori_distance_map']
            angle_map = metas[i]['ori_angle_map']
            seg_label_map = metas[i]['ori_target']

            gt_mask_label = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map)

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            ori_target = metas[i]['ori_target']

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            dir_logits = cv2.resize(dir_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)                                      
            mask_pred = np.argmax(mask_logits, axis=-1)
            dir_pred = np.argmax(dir_logits, axis=-1)

            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map,
                extra_ignore_mask=mask_pred == 0
            )
          
            self.running_score[0].update(mask_pred[None], gt_mask_label[None])

            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map, 
                extra_ignore_mask=mask_pred == 0
            ) 
            self.running_score[1].update(dir_pred[None], gt_dir_label[None])

            dir_pred[mask_pred == 0] = DTOffsetConfig.num_classes
            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map, 
                extra_ignore_mask=gt_mask_label == 0,
            )
            self.running_score[2].update(dir_pred[None], gt_dir_label[None])
            # self._print_running_score()         

    def _get_multilabel_prediction(self, dir_logits, no_offset_mask=None, topk=8):
        mode = os.environ.get('ml_dt_offset_eval_mode', 'weighted_avg')

        if mode == 'max':
            dir_pred = dir_logits.argmax(axis=-1)
            if no_offset_mask is not None:
                dir_pred[no_offset_mask] = 8
            return dir_pred

        from lib.utils.helpers.offset_helper import DTOffsetHelper
        h, w, _ = dir_logits.shape
        dir_logits = torch.from_numpy(dir_logits).unsqueeze(0).permute(0, 3, 1, 2)
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

    def _update_multi_label_dt_offset_score(self, outputs, metas):
        from lib.utils.helpers.offset_helper import DTOffsetHelper, DTOffsetConfig
        mask_maps, dir_maps = map(lambda x: x.permute(0, 2, 3, 1), outputs)

        if 'relax' in self.configer.get('loss', 'loss_type'):
            dir_maps = torch.softmax(dir_maps, dim=-1)
        else:
            dir_maps = torch.sigmoid(dir_maps)

        for i in range(mask_maps.size(0)):

            distance_map = metas[i]['ori_distance_map']
            seg_label_map = metas[i]['ori_target']

            gt_mask_label = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map)

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            ori_target = metas[i]['ori_target']
            dir_label_map = metas[i]['ori_multi_label_direction_map']

            dir_label_map = DTOffsetHelper.encode_multi_labels(dir_label_map)
            dir_label_map[seg_label_map == -1, :] = -1

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            dir_logits = cv2.resize(dir_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)                                      
            mask_pred = np.argmax(mask_logits, axis=-1)

            dir_pred = self._get_multilabel_prediction(
                dir_logits, 
                no_offset_mask=mask_pred == 0,
                topk=8
            )

            self.running_score[0].update(mask_pred[None], gt_mask_label[None])
            self.running_score[1].update(dir_pred, dir_label_map, (mask_pred == 1) & (seg_label_map != -1))
            self.running_score[2].update(dir_pred, dir_label_map, gt_mask_label == 1)

            # self._print_running_score()

    def _update_sobel_mask_dt_offset_score(self, outputs, metas):
        from lib.utils.helpers.offset_helper import DTOffsetHelper, DTOffsetConfig
        mask_maps, _ = map(lambda x: x.permute(0, 2, 3, 1), outputs)

        for i in range(mask_maps.size(0)):

            distance_map = metas[i]['ori_distance_map']
            angle_map = metas[i]['ori_angle_map']
            seg_label_map = metas[i]['ori_target']

            gt_mask_label = DTOffsetHelper.distance_to_mask_label(distance_map, seg_label_map)

            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            ori_target = metas[i]['ori_target']

            mask_logits = cv2.resize(mask_maps[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
 
            mask_pred = np.argmax(mask_logits, axis=-1)

            dir_pred = DTOffsetHelper.edge_mask_to_vector(
                torch.softmax(
                    torch.from_numpy(mask_logits).unsqueeze(0).permute(0, 3, 1, 2),
                    dim=1,
                )[:, 1:2, :, :],
                normalized=True
            ).squeeze(0).permute(1, 2, 0).numpy()
            dir_pred = DTOffsetHelper.vector_to_label(dir_pred)

            self.running_score[0].update(mask_pred[None], gt_mask_label[None])

            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map, 
                extra_ignore_mask=mask_pred == 0
            ) 
            self.running_score[1].update(dir_pred[None], gt_dir_label[None])

            dir_pred[mask_pred == 0] = DTOffsetConfig.num_classes
            gt_dir_label = DTOffsetHelper.angle_to_direction_label(
                angle_map, 
                seg_label_map=seg_label_map, 
                extra_ignore_mask=gt_mask_label == 0,
            )
            self.running_score[2].update(dir_pred[None], gt_dir_label[None])      

    def _update_offset_score(self, pred_h, pred_w, metas, class_id=None, direction='h', idx=0):
        # compute the offset accuracy along the h or w direction for the given category as class_id
        pred_h = pred_h.permute(0, 2, 3, 1)
        pred_w = pred_w.permute(0, 2, 3, 1)
        for i in range(pred_h.size(0)):
            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            ori_target = metas[i]['ori_target']
            gt_offset_h = metas[i]['ori_offset_h']
            gt_offset_w = metas[i]['ori_offset_w']

            gt_offset_h = np.sign(gt_offset_h)
            gt_offset_w = np.sign(gt_offset_w)
            gt_offset_h[gt_offset_h == -1] = 2
            gt_offset_w[gt_offset_w == -1] = 2

            offset_h_logits = cv2.resize(pred_h[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            offset_w_logits = cv2.resize(pred_w[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)

            offset_h_logits = np.argmax(offset_h_logits, axis=-1)
            offset_w_logits = np.argmax(offset_w_logits, axis=-1)

            if class_id is not None:
                mask = ori_target != class_id
                offset_h_logits[mask] = 0
                offset_w_logits[mask] = 0
                gt_offset_h[mask] = 0
                gt_offset_w[mask] = 0

            if direction == 'h':
                self.running_score[idx].update(offset_h_logits[None], gt_offset_h[None])
            else:
                self.running_score[idx].update(offset_w_logits[None], gt_offset_w[None])  

    def train(self):
        # cudnn.benchmark = True
        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_val'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
            if self.configer.get('network', 'resume_train'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))

        while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):
            self.__train()

        # use swa to average the model
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
        self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))


if __name__ == "__main__":
    # Test class for pose estimator.
    pass