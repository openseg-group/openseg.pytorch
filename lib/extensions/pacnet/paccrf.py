"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

from lib.extensions.pacnet import pac


def create_position_feats(shape, scales=None, bs=1, device=None):
    cord_range = [range(s) for s in shape]
    mesh = np.array(np.meshgrid(*cord_range, indexing='ij'), dtype=np.float32)
    mesh = th.from_numpy(mesh)
    if device is not None:
        mesh = mesh.to(device)
    if scales is not None:
        if not isinstance(scales, th.Tensor):
            scales = th.tensor(scales, dtype=th.float32, device=device)
        mesh = mesh * (1.0 / scales.view(-1, 1, 1))
    return th.stack(bs * [mesh])


def create_YXRGB(img, yx_scale=None, rgb_scale=None, scales=None):
    img = img.view(-1, *img.shape[-3:])
    if scales is not None:
        assert yx_scale == None and rgb_scale == None
        yx_scale = scales[:2]
        rgb_scale = scales[2:]
    mesh = create_position_feats(img.shape[-2:], yx_scale, img.shape[0], img.device)
    if rgb_scale is not None:
        if not isinstance(rgb_scale, th.Tensor):
            rgb_scale = th.tensor(rgb_scale, dtype=th.float32, device=img.device)
        img = img * (1.0 / rgb_scale.view(-1, 1, 1))
    feats = th.cat([mesh, img], dim=1)
    return feats


def _ceil_pad_factor(sizes, factor):
    offs = tuple((factor - sz % factor) % factor for sz in sizes)
    pad = tuple((off + 1) // 2 for off in offs)
    return pad


class PacCRF(nn.Module):
    r"""
    Args:
        channels (int): number of categories.
        num_steps (int): number of mean-field update steps.
        final_output (str): 'log_softmax' | 'softmax' | 'log_Q'. Default: 'log_Q'
        perturbed_init (bool): whether to perturb initialization. Default: True
        native_impl (bool): Default: False
        fixed_weighting (bool): whether to use fixed weighting for unary/pairwise terms. Default: False
        unary_weight (float): Default: 1.0
        pairwise_kernels (dict or list): pairwise kernels, see add_pairwise_kernel() for details. Default: None
    """
    def __init__(self, channels, num_steps, final_output='log_Q', perturbed_init=True, native_impl=False,
                 fixed_weighting=False, unary_weight=1.0, pairwise_kernels=None):
        super(PacCRF, self).__init__()
        self.channels = channels
        self.num_steps = num_steps
        self.final_output = final_output  # 'log_softmax', 'softmax', 'log_Q'
        self.perturbed_init = perturbed_init
        self.native_impl = native_impl
        self.fixed_weighting = fixed_weighting
        self.init_unary_weight = unary_weight

        self.messengers = nn.ModuleList()
        self.compat = nn.ModuleList()
        self.init_pairwise_weights = []
        self.pairwise_weights = nn.ParameterList()
        self._use_pairwise_weights = []
        self.unary_weight = unary_weight if self.fixed_weighting else nn.Parameter(th.tensor(float(unary_weight)))
        self.blur = []
        self.pairwise_repr = []

        if pairwise_kernels is not None:
            if type(pairwise_kernels) == dict:
                self.add_pairwise_kernel(**pairwise_kernels)
            else:
                for k in pairwise_kernels:
                    self.add_pairwise_kernel(**k)

    def reset_parameters(self, pairwise_idx=None):
        if pairwise_idx is None:
            idxs = range(len(self.messengers))
            if not self.fixed_weighting:
                self.unary_weight.data.fill_(self.init_unary_weight)
        else:
            idxs = [pairwise_idx]

        for i in idxs:
            self.messengers[i].reset_parameters()
            if isinstance(self.messengers[i], nn.Conv2d):
                # TODO: gaussian initialization for XY kernels?
                pass
            if self.compat[i] is not None:
                self.compat[i].weight.data[:, :, 0, 0] = 1.0 - th.eye(self.channels, dtype=th.float32)
                if self.perturbed_init:
                    perturb_range = 0.001
                    self.compat[i].weight.data.add_((th.rand_like(self.compat[i].weight.data) - 0.5) * perturb_range)
            self.pairwise_weights[i].data = th.ones_like(self.pairwise_weights[i]) * self.init_pairwise_weights[i]

    def extra_repr(self):
        s = ('categories={channels}'
             ', num_steps={num_steps}'
             ', final_output={final_output}')
        if self.perturbed_init:
            s += ', perturbed_init=True'
        if self.fixed_weighting:
            s += ', fixed_weighting=True'
        if self.pairwise_repr:
            s += ', pairwise_kernels=({})'.format(', '.join(self.pairwise_repr))
        return s.format(**self.__dict__)

    def add_pairwise_kernel(self, kernel_size=3, dilation=1, blur=1, compat_type='4d', spatial_filter=True,
                            pairwise_weight=1.0):
        assert kernel_size % 2 == 1
        self.pairwise_repr.append('{}{}_{}_{}_{}'.format('0d' if compat_type == 'potts' else compat_type,
                                                         's' if spatial_filter else '',
                                                         kernel_size, dilation, blur))

        if compat_type == 'potts':
            pairwise_weight *= -1.0

        if compat_type == 'potts' and (not spatial_filter) and (not self.fixed_weighting):
            self._use_pairwise_weights.append(True)
        else:
            self._use_pairwise_weights.append(False)
        self.pairwise_weights.append(nn.Parameter(th.tensor(pairwise_weight, dtype=th.float32)))
        self.init_pairwise_weights.append(pairwise_weight)
        self.blur.append(blur)
        self.compat.append(nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False) if compat_type == '2d'
                           else None)

        pad = int(kernel_size // 2) * dilation

        if compat_type == 'na':
            messenger = nn.Conv2d(self.channels, self.channels, kernel_size, padding=pad, dilation=dilation, bias=False)
        elif compat_type == '4d':
            messenger = pac.PacConv2d(self.channels, self.channels, kernel_size, padding=pad, dilation=dilation,
                                          bias=False, shared_filters=False, native_impl=self.native_impl,
                                          filler=('crf_perturbed' if self.perturbed_init else 'crf'))
        elif spatial_filter:
            messenger = pac.PacConv2d(self.channels, self.channels, kernel_size, padding=pad, dilation=dilation,
                                          bias=False, shared_filters=True, native_impl=self.native_impl,
                                          filler=('crf_perturbed' if self.perturbed_init else 'crf'))
        else:
            messenger = pac.PacConv2d(self.channels, self.channels, kernel_size, padding=pad, dilation=dilation,
                                          bias=False, shared_filters=True, native_impl=self.native_impl,
                                          filler='crf_pool')

        self.messengers.append(messenger)
        self.reset_parameters(-1)

    def num_pairwise_kernels(self):
        return len(self.messengers)

    def forward(self, unary, edge_feat, edge_kernel=None, logQ=None):
        n_kernels = len(self.messengers)
        edge_kernel = [edge_kernel] * n_kernels if isinstance(edge_kernel, th.Tensor) else edge_kernel

        if edge_kernel is None:
            edge_kernel = [None] * n_kernels
            _shared = isinstance(edge_feat, th.Tensor)
            if _shared:
                edge_feat = {1 : edge_feat}
            for i in range(n_kernels):
                if isinstance(self.messengers[i], nn.Conv2d):
                    continue
                if _shared and self.blur[i] in edge_feat:
                    feat = edge_feat[self.blur[i]]
                elif self.blur[i] == 1:
                    feat = edge_feat[i]
                else:
                    feat = edge_feat[1] if _shared else edge_feat[i]
                    pad = _ceil_pad_factor(feat.shape[2:], self.blur[i])
                    feat = F.avg_pool2d(feat,
                                        kernel_size=self.blur[i],
                                        padding=pad,
                                        count_include_pad=False)
                    if _shared:
                        edge_feat[self.blur[i]] = feat
                edge_kernel[i], _ = self.messengers[i].compute_kernel(feat)
                del feat
            del edge_feat

        if logQ is None:
            logQ = unary
        for step in range(self.num_steps):
            Q = F.softmax(logQ, dim=1)
            Q_blur = {1 : Q}
            logQ = unary * self.unary_weight
            for i in range(n_kernels):
                pad = _ceil_pad_factor(Q.shape[2:], self.blur[i])
                if self.blur[i] not in Q_blur:
                    Q_blur[self.blur[i]] = F.avg_pool2d(Q,
                                                        kernel_size=self.blur[i],
                                                        padding=pad,
                                                        count_include_pad=False)
                if isinstance(self.messengers[i], nn.Conv2d):
                    msg = self.messengers[i](Q_blur[self.blur[i]])
                else:
                    msg = self.messengers[i](Q_blur[self.blur[i]], None, edge_kernel[i])
                if self.compat[i] is not None:
                    msg = self.compat[i](msg)
                if self.blur[i] > 1:
                    msg = F.interpolate(msg, scale_factor=self.blur[i], mode='bilinear', align_corners=False)
                    msg = msg[:, :, pad[0]:pad[0] + unary.shape[2], pad[1]:pad[1] + unary.shape[3]].contiguous()
                pw = self.pairwise_weights[i] if self._use_pairwise_weights[i] else self.init_pairwise_weights[i]
                logQ = logQ - msg * pw

        if self.final_output == 'softmax':
            out = F.softmax(logQ, dim=1)
        elif self.final_output == 'log_softmax':
            out = F.log_softmax(logQ, dim=1)
        elif self.final_output == 'log_Q':
            out = logQ
        else:
            raise ValueError('Unknown value for final_output: {}'.format(self.final_output))

        return out


class PacCRFLoose(nn.Module):
    def __init__(self, channels, num_steps, final_output='log_Q', perturbed_init=True, native_impl=False,
                 fixed_weighting=False, unary_weight=1.0, pairwise_kernels=None):
        super(PacCRFLoose, self).__init__()
        self.channels = channels
        self.num_steps = num_steps
        self.final_output = final_output  # 'log_softmax', 'softmax', 'log_Q'

        self.steps = nn.ModuleList()
        for i in range(num_steps):
            self.steps.append(PacCRF(channels, 1, 'log_Q', perturbed_init, native_impl, fixed_weighting, unary_weight,
                                     pairwise_kernels))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_steps):
            self.steps[i].reset_parameters()

    def extra_repr(self):
        s = ('categories={channels}'
             ', num_steps={num_steps}'
             ', final_output={final_output}')
        return s.format(**self.__dict__)

    def add_pairwise_kernel(self, kernel_size=3, dilation=1, blur=1, compat_type='4d', spatial_filter=True,
                            pairwise_weight=1.0):
        for i in range(self.num_steps):
            self.steps[i].add_pairwise_kernel(kernel_size, dilation, blur, compat_type, spatial_filter, pairwise_weight)

    def num_pairwise_kernels(self):
        return self.steps[0].num_pairwise_kernels()

    def forward(self, unary, edge_feat, edge_kernel=None):
        n_kernels = self.num_pairwise_kernels()
        edge_kernel = [edge_kernel] * n_kernels if isinstance(edge_kernel, th.Tensor) else edge_kernel
        blurs = self.steps[0].blur

        if edge_kernel is None:
            edge_kernel = [None] * n_kernels
            _shared = isinstance(edge_feat, th.Tensor)
            if _shared:
                edge_feat = {1 : edge_feat}
            for i in range(n_kernels):
                if _shared and blurs[i] in edge_feat:
                    feat = edge_feat[blurs[i]]
                elif blurs[i] == 1:
                    feat = edge_feat[i]
                else:
                    feat = edge_feat[1] if _shared else edge_feat[i]
                    pad = _ceil_pad_factor(feat.shape[2:], blurs[i])
                    feat = F.avg_pool2d(feat,
                                        kernel_size=blurs[i],
                                        padding=pad,
                                        count_include_pad=False)
                    if _shared:
                        edge_feat[blurs[i]] = feat
                edge_kernel[i], _ = self.steps[0].messengers[i].compute_kernel(feat)
                del feat
            del edge_feat

        logQ = unary
        for step in self.steps:
            logQ = step(unary, None, edge_kernel, logQ)

        if self.final_output == 'softmax':
            out = F.softmax(logQ, dim=1)
        elif self.final_output == 'log_softmax':
            out = F.log_softmax(logQ, dim=1)
        elif self.final_output == 'log_Q':
            out = logQ
        else:
            raise ValueError('Unknown value for final_output: {}'.format(self.final_output))

        return out
