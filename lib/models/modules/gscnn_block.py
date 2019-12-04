"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import numpy as np
import math

from lib.models.tools.module_helper import ModuleHelper

torch_ver = torch.__version__[:3]


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, configer, in_dim, reduction_dim=256, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        self.configer = configer

        if "dilated8" in self.configer.get("network", "backbone"):
            output_stride = 8
        elif "dilated16" in self.configer.get("network", "backbone"):
            output_stride = 16
        else:
            output_stride = None

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise RuntimeError(
                "output stride of {} not supported".format(output_stride)
            )

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                ModuleHelper.BNReLU(
                    reduction_dim, bn_type=self.configer.get("network", "bn_type")
                ),
            )
        )
        # other rates
        for r in rates:
            self.features.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_dim,
                        reduction_dim,
                        kernel_size=3,
                        dilation=r,
                        padding=r,
                        bias=False,
                    ),
                    ModuleHelper.BNReLU(
                        reduction_dim, bn_type=self.configer.get("network", "bn_type")
                    ),
                )
            )
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(
                reduction_dim, bn_type=self.configer.get("network", "bn_type")
            ),
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(
                reduction_dim, bn_type=self.configer.get("network", "bn_type")
            ),
        )

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(
            img_features, x_size[2:], mode="bilinear", align_corners=True
        )
        out = img_features

        edge_features = F.interpolate(
            edge, x_size[2:], mode="bilinear", align_corners=True
        )
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class GatedSpatialConv2d(_ConvNd):
    def __init__(
        self,
        configer,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """
        self.configer = configer

        if torch_ver == "0.4":
            extra_args = []
        else:
            extra_args = ["zeros"]

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            *extra_args,
        )

        self._gate_conv = nn.Sequential(
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get("network", "bn_type"))(
                in_channels + 1
            ),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get("network", "bn_type"))(
                1
            ),
            nn.Sigmoid(),
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = input_features * (alphas + 1)
        return F.conv2d(
            input_features,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Conv2dPad(nn.Conv2d):
    def forward(self, input):
        return myF.conv2d_same(input, self.weight, self.groups)


class HighFrequencyGatedSpatialConv2d(_ConvNd):
    def __init__(
        self,
        configer,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(HighFrequencyGatedSpatialConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
        )

        self._gate_conv = nn.Sequential(
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get("network", "bn_type"))(
                in_channels + 1
            ),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get("network", "bn_type"))(
                1
            ),
            nn.Sigmoid(),
        )

        kernel_size = 7
        sigma = 3

        x_cord = torch.arange(kernel_size).float()
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size).float()
        y_grid = x_grid.t().float()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.0
        variance = sigma ** 2.0
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(in_channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=3,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False,
        )

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

        self.cw = nn.Conv2d(in_channels * 2, in_channels, 1)

        self.procdog = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            ModuleHelper.BatchNorm2d(bn_type=self.configer.get("network", "bn_type"))(
                in_channels
            ),
            nn.Sigmoid(),
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        n, c, h, w = input_features.size()
        smooth_features = self.gaussian_filter(input_features)
        dog_features = input_features - smooth_features
        dog_features = self.cw(torch.cat((dog_features, input_features), dim=1))

        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        dog_features = dog_features * (alphas + 1)

        return F.conv2d(
            dog_features,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def test():
    import matplotlib.pyplot as plt

    canny_map_filters_in = 8
    canny_map = np.random.normal(size=(1, canny_map_filters_in, 10, 10))  # NxCxHxW
    resnet_map = np.random.normal(size=(1, 1, 10, 10))  # NxCxHxW
    plt.imshow(canny_map[0, 0])
    plt.show()

    canny_map = torch.from_numpy(canny_map).float()
    resnet_map = torch.from_numpy(resnet_map).float()

    gconv = GatedSpatialConv2d(
        canny_map_filters_in, canny_map_filters_in, kernel_size=3, stride=1, padding=1
    )
    output_map = gconv(canny_map, resnet_map)
    print("done")


if __name__ == "__main__":
    test()