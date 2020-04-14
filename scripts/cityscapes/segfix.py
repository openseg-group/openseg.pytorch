#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import scipy.io as io
import subprocess
import multiprocessing.pool as mpp

DATA_ROOT = subprocess.check_output(
    ['bash', '-c', "source config.profile; echo $DATA_ROOT"]
).decode().strip()

import os
import sys
import argparse
import os.path as osp

script_path = osp.abspath(osp.join(osp.dirname(__file__)))
os.chdir(osp.join(script_path, '..', '..'))
sys.path.insert(0, os.getcwd())
os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')
from lib.models.modules.offset_block import OffsetModule

class LabelTransformer:

    label_list = [7, 8, 11, 12, 13, 17, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    @staticmethod
    def encode(labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(
            shape=(shape[0], shape[1]), dtype=np.int) * 255
        for i in range(len(LabelTransformer.label_list)):
            class_id = LabelTransformer.label_list[i]
            encoded_labelmap[labelmap == class_id] = i

        return encoded_labelmap

    @staticmethod
    def decode(labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(
            shape=(shape[0], shape[1]), dtype=np.uint8) * 255
        for i in range(len(LabelTransformer.label_list)):
            class_id = i
            encoded_labelmap[labelmap ==
                             class_id] = LabelTransformer.label_list[i]

        return encoded_labelmap


def gen_coord_map(H, W):
    coord_vecs = [torch.arange(length, dtype=torch.float) for length in (H, W)]
    coord_h, coord_w = torch.meshgrid(coord_vecs)
    return coord_h, coord_w


def get_onehot(label_map, num_classes=19):
    onehot = torch.eye(num_classes)
    return onehot[label_map.long()].permute(0, 3, 1, 2)


def shift(x, offset):
    """
    x: h x w
    offset: 2 x h x w
    """
    h, w = x.shape
    x = torch.from_numpy(x).unsqueeze(0)
    onehot = get_onehot(x)
    offset = torch.from_numpy(offset).unsqueeze(0)
    coord_map = gen_coord_map(h, w)
    norm_factor = torch.FloatTensor([(w-1)/2, (h-1)/2])
    grid_h = offset[:, 0]+coord_map[0]
    grid_w = offset[:, 1]+coord_map[1]
    grid = torch.stack([grid_w, grid_h], dim=-1) / norm_factor - 1
    onehot = F.grid_sample(onehot, grid, padding_mode='border', mode='bilinear')
    return onehot.argmax(dim=1).squeeze().numpy().astype(np.uint8)

def get_offset(basename):
    return io.loadmat(osp.join(offset_dir, basename+'.mat'))['mat']\
        .astype(np.float32).transpose(2, 0, 1) * args.scale

def process(basename):
    infile = osp.join(in_label_dir, basename + '.png')
    outfile = osp.join(out_label_dir, basename + '.png')

    input_label_map = np.array(Image.open(infile).convert('P'))
    input_label_map = LabelTransformer.encode(input_label_map)

    offset_map = get_offset(basename)
    output_label_map = shift(input_label_map, offset_map)
    output_label_map = LabelTransformer.decode(output_label_map)
    Image.fromarray(output_label_map).save(outfile)
    print('Writing', outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--offset')
    parser.add_argument('--out')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--scale', type=float, default=2)
    args = parser.parse_args()

    offset_dir = args.offset
    in_label_dir = args.input
    if args.out is None:
        out_label_dir = in_label_dir.replace('/label', '/label_w_segfix')
    else:
        out_label_dir = args.out
    print('Saving to', out_label_dir)

    os.makedirs(out_label_dir, exist_ok=True)
    input_args = [fn.rpartition('.')[0] for fn in os.listdir(in_label_dir)]
    print(len(input_args), 'files in total.')
    mpp.Pool().map(process, input_args)

    if args.split == 'val':
        os.system('{} lib/metrics/cityscapes_evaluator.py --gt_dir {}/cityscapes/val/label --pred_dir {}'.format(sys.executable, DATA_ROOT, out_label_dir))