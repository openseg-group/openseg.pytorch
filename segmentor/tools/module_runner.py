#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel.scatter_gather import gather as torch_gather

from lib.extensions.parallel.data_parallel import DataParallelModel
from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import get_rank, is_distributed

class ModuleRunner(object):

    def __init__(self, configer):
        self.configer = configer
        self._init()

    def _init(self):
        self.configer.add(['iters'], 0)
        self.configer.add(['last_iters'], 0)
        self.configer.add(['epoch'], 0)
        self.configer.add(['last_epoch'], 0)
        self.configer.add(['max_performance'], 0.0)
        self.configer.add(['performance'], 0.0)
        self.configer.add(['min_val_loss'], 9999.0)
        self.configer.add(['val_loss'], 9999.0)
        if not self.configer.exists('network', 'bn_type'):
            self.configer.add(['network', 'bn_type'], 'torchbn')

        if self.configer.get('phase') == 'train':
            assert len(self.configer.get('gpu')) > 1 or self.configer.get('network', 'bn_type') == 'torchbn'

        Log.info('BN Type is {}.'.format(self.configer.get('network', 'bn_type')))

    def to_device(self, *params, force_list=False):
        if is_distributed():
            device = torch.device('cuda:{}'.format(get_rank()))
        else:
            device = torch.device('cpu' if self.configer.get('gpu') is None else 'cuda')
        return_list = list()
        for i in range(len(params)):
            return_list.append(params[i].to(device))

        if force_list:
            return return_list
        else:
            return return_list[0] if len(params) == 1 else return_list

    def _make_parallel(self, net):
        if is_distributed():
            local_rank = get_rank()

            return torch.nn.parallel.DistributedDataParallel(
                net,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        if len(self.configer.get('gpu')) == 1:
            self.configer.update(['network', 'gathered'], True)

        return DataParallelModel(net, gather_=self.configer.get('network', 'gathered'))

    def load_net(self, net):
        net = self.to_device(net)
        net = self._make_parallel(net)

        if not is_distributed():
            net = net.to(torch.device('cpu' if self.configer.get('gpu') is None else 'cuda'))

        net.float()
        if self.configer.get('network', 'resume') is not None:
            Log.info('Loading checkpoint from {}...'.format(self.configer.get('network', 'resume')))
            resume_dict = torch.load(self.configer.get('network', 'resume'))
            if 'state_dict' in resume_dict:
                checkpoint_dict = resume_dict['state_dict']

            elif 'model' in resume_dict:
                checkpoint_dict = resume_dict['model']

            elif isinstance(resume_dict, OrderedDict):
                checkpoint_dict = resume_dict

            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(self.configer.get('network', 'resume')))

            if list(checkpoint_dict.keys())[0].startswith('module.'):
                checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items()}

            # load state_dict
            if hasattr(net, 'module'):
                self.load_state_dict(net.module, checkpoint_dict, self.configer.get('network', 'resume_strict'))
            else:
                self.load_state_dict(net, checkpoint_dict, self.configer.get('network', 'resume_strict'))

            if self.configer.get('network', 'resume_continue'):
                self.configer.resume(resume_dict['config_dict'])

        return net

    @staticmethod
    def load_state_dict(module, state_dict, strict=False):
        """Load state_dict to a module.
        This method is modified from :meth:`torch.nn.Module.load_state_dict`.
        Default value for ``strict`` is set to ``False`` and the message for
        param mismatch will be shown even if strict is False.
        Args:
            module (Module): Module that receives the state_dict.
            state_dict (OrderedDict): Weights.
            strict (bool): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        """
        unexpected_keys = []
        own_state = module.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                own_state[name].copy_(param)
            except Exception:
                Log.warn('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(),
                                           param.size()))
                
        missing_keys = set(own_state.keys()) - set(state_dict.keys())

        err_msg = []
        if unexpected_keys:
            err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
        if missing_keys:
            # we comment this to fine-tune the models with some missing keys.
            err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
        err_msg = '\n'.join(err_msg)
        if err_msg:
            if strict:
                raise RuntimeError(err_msg)
            else:
                Log.warn(err_msg)

    def save_net(self, net, save_mode='iters'):
        if is_distributed() and get_rank() != 0:
            return

        state = {
            'config_dict': self.configer.to_dict(),
            'state_dict': net.state_dict(),
        }
        if self.configer.get('checkpoints', 'checkpoints_root') is None:
            checkpoints_dir = os.path.join(self.configer.get('project_dir'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))
        else:
            checkpoints_dir = os.path.join(self.configer.get('checkpoints', 'checkpoints_root'),
                                           self.configer.get('checkpoints', 'checkpoints_dir'))

        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        latest_name = '{}_latest.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
        torch.save(state, os.path.join(checkpoints_dir, latest_name))
        if save_mode == 'performance':
            if self.configer.get('performance') > self.configer.get('max_performance'):
                latest_name = '{}_max_performance.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['max_performance'], self.configer.get('performance'))

        elif save_mode == 'val_loss':
            if self.configer.get('val_loss') < self.configer.get('min_val_loss'):
                latest_name = '{}_min_loss.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['min_val_loss'], self.configer.get('val_loss'))

        elif save_mode == 'iters':
            if self.configer.get('iters') - self.configer.get('last_iters') >= \
                    self.configer.get('checkpoints', 'save_iters'):
                latest_name = '{}_iters{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('iters'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['last_iters'], self.configer.get('iters'))

        elif save_mode == 'epoch':
            if self.configer.get('epoch') - self.configer.get('last_epoch') >= \
                    self.configer.get('checkpoints', 'save_epoch'):
                latest_name = '{}_epoch{}.pth'.format(self.configer.get('checkpoints', 'checkpoints_name'),
                                                 self.configer.get('epoch'))
                torch.save(state, os.path.join(checkpoints_dir, latest_name))
                self.configer.update(['last_epoch'], self.configer.get('epoch'))

        else:
            Log.error('Metric: {} is invalid.'.format(save_mode))
            exit(1)

    def freeze_bn(self, net, syncbn=False):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

            if syncbn:
                from lib.extensions import BatchNorm2d, BatchNorm1d
                if isinstance(m, BatchNorm2d) or isinstance(m, BatchNorm1d):
                    m.eval()

    def clip_grad(self, model, max_grad=10.):
        """Computes a gradient clipping coefficient based on gradient norm."""
        total_norm = 0
        for p in model.parameters():
            if p.requires_grad:
                modulenorm = p.grad.data.norm()
                total_norm += modulenorm ** 2

        total_norm = math.sqrt(total_norm)

        norm = max_grad / max(total_norm, max_grad)
        for p in model.parameters():
            if p.requires_grad:
                p.grad.mul_(norm)

    def gather(self, outputs, target_device=None, dim=0):
        r"""
        Gathers tensors from different GPUs on a specified device
          (-1 means the CPU).
        """
        if not self.configer.get('network', 'gathered'):
            if target_device is None:
                target_device = list(range(torch.cuda.device_count()))[0]

            return torch_gather(outputs, target_device, dim=dim)

        else:
            return outputs

    def get_lr(self, optimizer):

        return [param_group['lr'] for param_group in optimizer.param_groups]

    def warm_lr(self, iters, scheduler, optimizer, backbone_list=(0, )):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if not self.configer.exists('lr', 'is_warm') or not self.configer.get('lr', 'is_warm'):
            return

        warm_iters = self.configer.get('lr', 'warm')['warm_iters']
        if iters < warm_iters:
            if self.configer.get('lr', 'warm')['freeze_backbone']:
                for backbone_index in backbone_list:
                    optimizer.param_groups[backbone_index]['lr'] = 0.0

            else:
                lr_ratio = (self.configer.get('iters') + 1) / warm_iters
                base_lr_list = scheduler.get_lr()
                for backbone_index in backbone_list:
                    optimizer.param_groups[backbone_index]['lr'] = base_lr_list[backbone_index] * (lr_ratio ** 4)

