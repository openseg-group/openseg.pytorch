"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import unittest
from functools import wraps

import numpy as np
import torch as th
from torch import nn
from torch.autograd import gradcheck

import pac


def _allclose(x1, x2, rtol=1e-5, atol=1e-10):
    return np.allclose(x1.cpu(), x2.cpu(), rtol=rtol, atol=atol)


def _gradcheck(f, x0, rtol=1e-3, atol=1e-8):
    return gradcheck(f, x0, rtol=rtol, atol=atol)


# test both native autograd version and Function version
def repeat_impl_types(f):
    @wraps(f)
    def call_wrapped(self, *args):
        f(self, *args, native_impl=True)
        f(self, *args, native_impl=False)

    return call_wrapped


# some features are not yet implemented using custom Function
def use_only_native_impl(f):
    @wraps(f)
    def call_wrapped(self, *args):
        f(self, *args, native_impl=True)

    return call_wrapped


# test only the version with custom Function
def use_only_custom_impl(f):
    @wraps(f)
    def call_wrapped(self, *args):
        f(self, *args, native_impl=False)

    return call_wrapped


class PacConvTest(unittest.TestCase):
    def setUp(self):
        self.device = th.device('cuda:0')
        th.cuda.set_device(0)

    @repeat_impl_types
    def test_conv_forward_const_kernel(self, native_impl):
        bs, sz, k_ch = 2, 111, 5
        args = dict(in_channels=4, out_channels=3, kernel_size=5, stride=2, padding=4, dilation=2)
        im = th.rand(bs, args['in_channels'], sz, sz).to(self.device)
        im_th = im.clone()
        im_k = th.ones(bs, k_ch, sz, sz).to(self.device)
        conv_w = th.rand(args['out_channels'], args['in_channels'],
                         args['kernel_size'], args['kernel_size']).to(self.device)
        conv_b = th.rand(args['out_channels']).to(self.device)
        conv = pac.PacConv2d(native_impl=native_impl, **args).to(self.device)
        conv_th = nn.Conv2d(**args).to(self.device)
        conv.weight.data[:] = conv_th.weight.data[:] = conv_w
        conv.bias.data[:] = conv_th.bias.data[:] = conv_b

        _allclose(conv(im, im_k).detach(), conv_th(im_th).detach())

    @repeat_impl_types
    def test_conv_transpose_forward_const_kernel(self, native_impl):
        bs, sz, k_ch = 4, 128, 5
        args = dict(in_channels=4, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1, dilation=1)
        k_with_d = (args['kernel_size'] - 1) * args['dilation'] + 1
        sz_out = (sz - 1) * args['stride'] - 2 * args['padding'] + k_with_d + args['output_padding']
        im = th.rand(bs, args['in_channels'], sz, sz).to(self.device)
        im_th = im.clone()
        im_k = th.ones(bs, k_ch, sz_out, sz_out).to(self.device)
        conv_w = th.rand(args['in_channels'], args['out_channels'],
                         args['kernel_size'], args['kernel_size']).to(self.device)
        conv_b = th.rand(args['out_channels']).to(self.device)
        conv = pac.PacConvTranspose2d(native_impl=native_impl, **args).to(self.device)
        conv_th = nn.ConvTranspose2d(**args).to(self.device)
        conv.weight.data[:] = conv_th.weight.data[:] = conv_w
        conv.bias.data[:] = conv_th.bias.data[:] = conv_b

        _allclose(conv(im, im_k).detach(), conv_th(im_th).detach())

    @repeat_impl_types
    def test_pool_forward_const_kernel(self, native_impl):
        bs, sz, in_ch, k_ch = 2, 9, 4, 5
        dilation = 1
        args = dict(kernel_size=5, stride=2, padding=2)
        im = th.rand(bs, in_ch, sz, sz).to(self.device)
        im_th = im.clone()
        im_k = th.ones(bs, k_ch, sz, sz).to(self.device)
        pool = pac.PacPool2d(dilation=dilation, native_impl=native_impl, **args).to(self.device)
        pool_th = nn.AvgPool2d(**args).to(self.device)

        _allclose(pool(im, im_k).detach(), pool_th(im_th).detach())

    @repeat_impl_types
    def test_conv_input_grad(self, native_impl):
        bs, sz, k_ch = 2, 8, 3
        args = dict(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1, dilation=1)
        im = th.rand(bs, args['in_channels'], sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz, sz).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True
        conv = pac.PacConv2d(native_impl=native_impl, **args).double().to(self.device)
        self.assertTrue(_gradcheck(conv, (im, im_k)))

    @use_only_native_impl
    def test_conv_inv_kernel_input_grad(self, native_impl):
        bs, sz, k_ch = 2, 8, 3
        args = dict(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1, dilation=1,
                    kernel_type='inv_0.2_0.2_asym', smooth_kernel_type='average_5', normalize_kernel=True)
        im = th.rand(bs, args['in_channels'], sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz, sz).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True
        conv = pac.PacConv2d(native_impl=native_impl, **args).double().to(self.device)
        self.assertTrue(_gradcheck(conv, (im, im_k)))

    @repeat_impl_types
    def test_conv_all_grad(self, native_impl):
        bs, sz, k_ch, f_sz, in_ch, out_ch = 2, 10, 3, 5, 2, 4
        conv_args = dict(stride=1, padding=2, dilation=2)
        kernel_args = dict(kernel_size=f_sz, smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                           kernel_type='gaussian', smooth_kernel_type='none',
                           channel_wise=False, normalize_kernel=False, transposed=False,
                           **conv_args)
        im = th.rand(bs, in_ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz, sz).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True

        conv_w = th.rand(out_ch, in_ch, f_sz, f_sz).double().to(self.device)
        conv_b = th.rand(out_ch).double().to(self.device)
        self.assertTrue(_gradcheck(
            lambda in0, in1, w, b: pac.pacconv2d(in0,
                                                 pac.packernel2d(in1, **kernel_args)[0],
                                                 w, b, native_impl=native_impl, **conv_args),
            (im, im_k, conv_w, conv_b)))

    @repeat_impl_types
    def test_conv_transpose_input_grad(self, native_impl):
        bs, sz, k_ch = 1, 4, 2
        args = dict(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        k_with_d = (args['kernel_size'] - 1) * args['dilation'] + 1
        sz_out = (sz - 1) * args['stride'] - 2 * args['padding'] + k_with_d + args['output_padding']
        im = th.rand(bs, args['in_channels'], sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz_out, sz_out).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True
        conv = pac.PacConvTranspose2d(native_impl=native_impl, **args).double().to(self.device)
        self.assertTrue(_gradcheck(conv, (im, im_k)))

    @repeat_impl_types
    def test_conv_transpose_all_grad(self, native_impl):
        bs, sz, k_ch, f_sz, in_ch, out_ch = 2, 3, 3, 3, 2, 3
        conv_args = dict(stride=2, padding=1, output_padding=1, dilation=1)
        kernel_args = dict(kernel_size=f_sz, smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                           kernel_type='gaussian', smooth_kernel_type='none',
                           channel_wise=False, normalize_kernel=False, transposed=True,
                           **conv_args)
        k_with_d = (f_sz - 1) * conv_args['dilation'] + 1
        sz_out = (sz - 1) * conv_args['stride'] - 2 * conv_args['padding'] + k_with_d + conv_args['output_padding']
        im = th.rand(bs, in_ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz_out, sz_out).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True

        conv_w = th.rand(in_ch, out_ch, f_sz, f_sz).double().to(self.device)
        conv_b = th.rand(out_ch).double().to(self.device)
        self.assertTrue(_gradcheck(
            lambda in0, in1, w, b: pac.pacconv_transpose2d(in0,
                                                           pac.packernel2d(in1, **kernel_args)[0],
                                                           w, b, native_impl=native_impl, **conv_args),
            (im, im_k, conv_w, conv_b)))

    @repeat_impl_types
    def test_pool_grad(self, native_impl):
        bs, sz, ch, k_ch = 2, 8, 2, 3
        args = dict(kernel_size=5, stride=2, padding=4, dilation=2)
        im = th.rand(bs, ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz, sz).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True
        pool = pac.PacPool2d(native_impl=native_impl, **args).double().to(self.device)
        self.assertTrue(_gradcheck(pool, (im, im_k)))

    def test_conv_two_impl_match(self):
        bs, sz, k_ch = 24, 128, 3
        args = dict(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=2, dilation=2)
        im = th.rand(bs, args['in_channels'], sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz, sz).double().to(self.device)
        im0 = im.clone()
        im0_k = im_k.clone()
        im.requires_grad = im_k.requires_grad = True
        im0.requires_grad = im0_k.requires_grad = True
        conv = pac.PacConv2d(native_impl=False, **args).double().to(self.device)
        conv0 = pac.PacConv2d(native_impl=True, **args).double().to(self.device)

        conv_w = th.rand(args['out_channels'], args['in_channels'],
                         args['kernel_size'], args['kernel_size']).double().to(self.device)
        conv_b = th.rand(args['out_channels']).double().to(self.device)
        conv.weight.data[:] = conv0.weight.data[:] = conv_w
        conv.bias.data[:] = conv0.bias.data[:] = conv_b

        out = conv(im, im_k)
        out0 = conv0(im0, im0_k)
        out.sum().backward()
        out0.sum().backward()
        self.assertTrue(_allclose(out.detach(), out0.detach()))
        self.assertTrue(_allclose(im.grad, im0.grad))
        self.assertTrue(_allclose(im_k.grad, im0_k.grad))
        self.assertTrue(_allclose(conv.weight.grad, conv0.weight.grad))
        self.assertTrue(_allclose(conv.bias.grad, conv0.bias.grad))

    def test_conv_with_kernel_input_two_impl_match(self):
        bs, sz, k_ch = 24, 128, 3
        args = dict(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=2, dilation=2)
        im = th.rand(bs, args['in_channels'], sz, sz).double().to(self.device)
        out_sz = int(np.floor(
            (sz + 2 * args['padding'] - (args['kernel_size'] - 1) * args['dilation'] - 1) / args['stride'])) + 1
        im_k = th.rand(bs, 1, args['kernel_size'], args['kernel_size'], out_sz, out_sz).double().to(self.device)
        im0 = im.clone()
        im0_k = im_k.clone()
        im.requires_grad = im_k.requires_grad = True
        im0.requires_grad = im0_k.requires_grad = True
        conv = pac.PacConv2d(native_impl=False, **args).double().to(self.device)
        conv0 = pac.PacConv2d(native_impl=True, **args).double().to(self.device)

        conv_w = th.rand(args['out_channels'], args['in_channels'],
                         args['kernel_size'], args['kernel_size']).double().to(self.device)
        conv_b = th.rand(args['out_channels']).double().to(self.device)
        conv.weight.data[:] = conv0.weight.data[:] = conv_w
        conv.bias.data[:] = conv0.bias.data[:] = conv_b

        out = conv(im, None, im_k)
        out0 = conv0(im0, None, im0_k)
        out.sum().backward()
        out0.sum().backward()
        self.assertTrue(_allclose(out.detach(), out0.detach()))
        self.assertTrue(_allclose(im.grad, im0.grad))
        self.assertTrue(_allclose(im_k.grad, im0_k.grad))
        self.assertTrue(_allclose(conv.weight.grad, conv0.weight.grad))
        self.assertTrue(_allclose(conv.bias.grad, conv0.bias.grad))

    def test_conv_transpose_two_impl_match(self):
        bs, sz, k_ch = 3, 128, 3
        args = dict(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1)
        k_with_d = (args['kernel_size'] - 1) * args['dilation'] + 1
        sz_out = (sz - 1) * args['stride'] - 2 * args['padding'] + k_with_d + args['output_padding']
        im = th.rand(bs, args['in_channels'], sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz_out, sz_out).double().to(self.device)
        im0 = im.clone()
        im0_k = im_k.clone()
        im.requires_grad = im_k.requires_grad = True
        im0.requires_grad = im0_k.requires_grad = True
        conv = pac.PacConvTranspose2d(native_impl=False, **args).double().to(self.device)
        conv0 = pac.PacConvTranspose2d(native_impl=True, **args).double().to(self.device)

        conv_w = th.rand(args['in_channels'], args['out_channels'],
                         args['kernel_size'], args['kernel_size']).double().to(self.device)
        conv_b = th.rand(args['out_channels']).double().to(self.device)
        conv.weight.data[:] = conv0.weight.data[:] = conv_w
        conv.bias.data[:] = conv0.bias.data[:] = conv_b

        out = conv(im, im_k)
        out0 = conv0(im0, im0_k)
        out.sum().backward()
        out0.sum().backward()
        self.assertTrue(_allclose(out.detach(), out0.detach()))
        self.assertTrue(_allclose(im.grad, im0.grad))
        self.assertTrue(_allclose(im_k.grad, im0_k.grad))
        self.assertTrue(_allclose(conv.weight.grad, conv0.weight.grad))
        self.assertTrue(_allclose(conv.bias.grad, conv0.bias.grad))

    def test_pool_two_impl_match(self):
        bs, sz, ch, k_ch = 2, 128, 4, 3
        args = dict(kernel_size=3, stride=2, padding=2, dilation=2)
        im = th.rand(bs, ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz, sz).double().to(self.device)
        im0 = im.clone()
        im0_k = im_k.clone()
        im.requires_grad = im_k.requires_grad = True
        im0.requires_grad = im0_k.requires_grad = True
        pool = pac.PacPool2d(native_impl=False, **args).to(self.device)
        p00l0 = pac.PacPool2d(native_impl=True, **args).to(self.device)

        out = pool(im, im_k)
        out0 = p00l0(im0, im0_k)
        out.sum().backward()
        out0.sum().backward()
        self.assertTrue(_allclose(out.detach(), out0.detach()))
        self.assertTrue(_allclose(im.grad, im0.grad))
        self.assertTrue(_allclose(im_k.grad, im0_k.grad))

    def test_kernel_two_impl_match(self):
        bs, sz, ch = 16, 256, 8
        args = dict(kernel_size=3, stride=1, padding=1, dilation=1)
        im = th.rand(bs, ch, sz, sz).double().to(self.device)
        im0 = im.clone()
        im.requires_grad = im0.requires_grad = True

        out = pac.packernel2d(im, native_impl=False, **args)[0]
        out0 = pac.packernel2d(im0, native_impl=True, **args)[0]

        out.sum().backward()
        out0.sum().backward()
        self.assertTrue(_allclose(out.detach(), out0.detach()))
        self.assertTrue(_allclose(im.grad, im0.grad))

    # Tests below pass on small input sizes, but may fail on larger ones

    @repeat_impl_types
    def test_conv_sum_all_grad(self, native_impl):
        bs, sz, k_ch, f_sz, in_ch, out_ch = 2, 10, 3, 5, 2, 4
        conv_args = dict(stride=1, padding=2, dilation=2)
        kernel_args = dict(kernel_size=f_sz, smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                           kernel_type='gaussian', smooth_kernel_type='none',
                           channel_wise=False, normalize_kernel=False, transposed=False,
                           **conv_args)
        im = th.rand(bs, in_ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz, sz).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True

        conv_w = th.rand(out_ch, in_ch, f_sz, f_sz).double().to(self.device)
        conv_b = th.rand(out_ch).double().to(self.device)
        self.assertTrue(_gradcheck(
            lambda in0, in1, w, b: pac.pacconv2d(in0,
                                                 pac.packernel2d(in1, **kernel_args)[0],
                                                 w, b, native_impl=native_impl, **conv_args).sum(),
            (im, im_k, conv_w, conv_b), rtol=0.01))

    @repeat_impl_types
    def test_conv_transpose_sum_all_grad(self, native_impl):
        bs, sz, k_ch, f_sz, in_ch, out_ch = 2, 3, 3, 3, 2, 3
        conv_args = dict(stride=2, padding=1, output_padding=1, dilation=1)
        kernel_args = dict(kernel_size=f_sz, smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                           kernel_type='gaussian', smooth_kernel_type='none',
                           channel_wise=False, normalize_kernel=False, transposed=True,
                           **conv_args)
        k_with_d = (f_sz - 1) * conv_args['dilation'] + 1
        sz_out = (sz - 1) * conv_args['stride'] - 2 * conv_args['padding'] + k_with_d + conv_args['output_padding']
        im = th.rand(bs, in_ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz_out, sz_out).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True

        conv_w = th.rand(in_ch, out_ch, f_sz, f_sz).double().to(self.device)
        conv_b = th.rand(out_ch).double().to(self.device)
        self.assertTrue(_gradcheck(
            lambda in0, in1, w, b: pac.pacconv_transpose2d(in0,
                                                           pac.packernel2d(in1, **kernel_args)[0],
                                                           w, b, native_impl=native_impl, **conv_args).sum(),
            (im, im_k, conv_w, conv_b), rtol=0.01))

    @repeat_impl_types
    def test_pool_sum_grad(self, native_impl):
        bs, sz, ch, k_ch = 2, 8, 2, 3
        args = dict(kernel_size=5, stride=2, padding=4, dilation=2)
        im = th.rand(bs, ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, k_ch, sz, sz).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True
        pool = pac.PacPool2d(native_impl=native_impl, **args).double().to(self.device)
        self.assertTrue(_gradcheck(lambda x, y: pool(x, y).sum(), (im, im_k), rtol=0.01))

    @repeat_impl_types
    def test_kernel_sum_grad(self, native_impl):
        bs, sz, ch = 2, 4, 4
        args = dict(kernel_size=3, stride=2, padding=1, dilation=1)
        im = th.rand(bs, ch, sz, sz).double().to(self.device)
        im.requires_grad = True
        self.assertTrue(_gradcheck(lambda x: pac.packernel2d(x, native_impl=native_impl, **args)[0].sum(),
                                   (im,), rtol=0.01))

    @repeat_impl_types
    def test_conv_with_kernel_input_sum_all_grad(self, native_impl):
        bs, sz, k_ch, f_sz, in_ch, out_ch = 2, 10, 3, 5, 2, 4
        args = dict(stride=1, padding=2, dilation=2)
        out_sz = int(np.floor((sz + 2 * args['padding'] - (f_sz - 1) * args['dilation'] - 1) / args['stride'])) + 1
        im = th.rand(bs, in_ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, 1, f_sz, f_sz, out_sz, out_sz).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True

        conv_w = th.rand(out_ch, in_ch, f_sz, f_sz).double().to(self.device)
        conv_b = th.rand(out_ch).double().to(self.device)
        self.assertTrue(_gradcheck(
            lambda in0, in1, w, b: pac.pacconv2d(in0, in1, w, b, native_impl=native_impl, **args).sum(),
            (im, im_k, conv_w, conv_b), rtol=0.01))

    @repeat_impl_types
    def test_conv_transpose_with_kernel_input_sum_all_grad(self, native_impl):
        bs, sz, k_ch, f_sz, in_ch, out_ch = 2, 3, 3, 3, 2, 3
        args = dict(stride=2, padding=1, output_padding=1, dilation=1)
        k_with_d = (f_sz - 1) * args['dilation'] + 1
        sz_out = (sz - 1) * args['stride'] - 2 * args['padding'] + k_with_d + args['output_padding']
        im = th.rand(bs, in_ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, 1, f_sz, f_sz, sz_out, sz_out).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True

        conv_w = th.rand(in_ch, out_ch, f_sz, f_sz).double().to(self.device)
        conv_b = th.rand(out_ch).double().to(self.device)
        self.assertTrue(_gradcheck(
            lambda in0, in1, w, b: pac.pacconv_transpose2d(in0, in1, w, b, native_impl=native_impl, **args).sum(),
            (im, im_k, conv_w, conv_b), rtol=0.01))

    @repeat_impl_types
    def test_pool_with_kernel_input_sum_grad(self, native_impl):
        bs, sz, ch = 2, 8, 2
        args = dict(kernel_size=3, stride=2, padding=2, dilation=2)
        out_sz = int(np.floor(
            (sz + 2 * args['padding'] - (args['kernel_size'] - 1) * args['dilation'] - 1) / args['stride'])) + 1
        im = th.rand(bs, ch, sz, sz).double().to(self.device)
        im_k = th.rand(bs, 1, args['kernel_size'], args['kernel_size'], out_sz, out_sz).double().to(self.device)
        im.requires_grad = im_k.requires_grad = True
        pool = pac.PacPool2d(native_impl=native_impl, **args).double().to(self.device)
        self.assertTrue(_gradcheck(lambda x, y: pool(x, None, y).sum(),
                                   (im, im_k), rtol=0.01))


if __name__ == '__main__':
    unittest.main()
