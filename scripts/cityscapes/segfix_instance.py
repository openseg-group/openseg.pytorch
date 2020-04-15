#!/usr/bin/env python3
"""
Author: Jingyi Xie (hsfzxjy@gmail.com)
"""

import multiprocessing.pool as mpp
import scipy.io as io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import glob
import shutil
import os.path as osp
import argparse
import sys
import time
import cv2

script_path = osp.abspath(osp.join(osp.dirname(__file__)))
os.chdir(osp.join(script_path, '..', '..'))
sys.path.insert(0, os.getcwd())
os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')

def gen_coord_map(H, W):
    coord_vecs = [torch.arange(length, dtype=torch.float) for length in (H, W)]
    coord_h, coord_w = torch.meshgrid(coord_vecs)
    return coord_h, coord_w


def shift(x, offset):
    """
    x: c x h x w
    offset: 2 x h x w
    """

    def do_shift(x, offset):
        grid_h = offset[:, 0] + coord_map[0]
        grid_w = offset[:, 1] + coord_map[1]
        grid = torch.stack([grid_w, grid_h], dim=-1) / norm_factor - 1

        x = F.grid_sample(
            x, grid, padding_mode='border', mode='bilinear')

        return x

    c, h, w = x.shape
    coord_map = gen_coord_map(h, w)
    norm_factor = torch.FloatTensor([(w-1)/2, (h-1)/2])

    x = torch.from_numpy(x).unsqueeze(0).float()

    offset = torch.from_numpy(offset).unsqueeze(
        0).clone() * args.scale
    x = do_shift(x, offset)

    return (x.squeeze(0).numpy() > 0.5).astype(np.uint8)


def get_offset(basename):
    return io.loadmat(osp.join(offset_dir, basename+'.mat'))['mat'].transpose(2, 0, 1).astype(np.float32)

def process(filename):
    infile = osp.join(in_dir, filename)
    print('Processing', infile)

    names = []
    masks = []
    with open(infile) as f:
        for line in f:
            name = line.strip().split()[0]
            names.append(name)
            mask = np.array(Image.open(osp.join(in_dir, name)).convert('P'))
            masks.append(mask)

    # Not that an image may have no instance prediction at all.
    if masks:
        masks = np.stack(masks, axis=0)
        masks = (masks > 0).astype(np.uint8)

        offset_map = get_offset(filename.replace('_pred.txt', ''))
        output_masks = shift(masks, offset_map)
    else:
        output_masks = []

    shutil.copy(infile, out_dir)
    for name, mask in zip(names, output_masks):
        out_name = osp.join(out_dir, name)
        Image.fromarray(
            mask * 255
        ).save(out_name)


def ensure_cityscapes_scripts():
    """
    Ensure that library `cityscapesscripts` is properly installed.

    Note that the original implementation from https://github.com/mcordts/cityscapesScripts will
    raise encoding error during installation. We then fork a copy for self-use.
    """
    try:
        import cityscapesscripts
    except ModuleNotFoundError:
        os.system(
            '{} -m pip install git+https://github.com/hsfzxjy/cityscapesScripts.git'.format(sys.executable))


def evaluation(pred_dir):
    """
    See https://github.com/facebookresearch/detectron2/blob/d250fcc1b66d5a3686c15144480441b7abe31dec/detectron2/evaluation/cityscapes_evaluation.py#L80
    """
    os.environ["CITYSCAPES_DATASET"] = args.dataset_dir
    ensure_cityscapes_scripts()
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval
    cityscapes_eval.args.predictionPath = pred_dir
    cityscapes_eval.args.predictionWalk = None
    cityscapes_eval.args.JSONOutput = False
    cityscapes_eval.args.colorized = False
    cityscapes_eval.args.gtInstancesFile = os.path.join(
        pred_dir, "gtInstances.json")

    groundTruthImgList = glob.glob(cityscapes_eval.args.groundTruthSearch)
    assert len(
        groundTruthImgList
    ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
        cityscapes_eval.args.groundTruthSearch
    )
    predictionImgList = []
    for gt in groundTruthImgList:
        predictionImgList.append(
            cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
    results = cityscapes_eval.evaluateImgLists(
        predictionImgList, groundTruthImgList, cityscapes_eval.args
    )["averages"]


def copy_gt():
    """
    Copy ground-truth information to output directory.

    The original file may use another dataset base dir, so we replace the keys 
    with currently used one.
    """
    if not osp.isfile(osp.join(in_dir, 'gtInstances.json')):
        return

    import json
    import re
    with open(osp.join(in_dir, 'gtInstances.json')) as f:
        content = json.load(f)
    new_content = {}
    target = osp.join(args.dataset_dir, 'gtFine')
    if target.endswith('/'):
        target = target[:-1]
    for key, value in content.items():
        key = re.sub(r'^.*?gtFine', target, key, 1)
        new_content[key] = value
    with open(osp.join(out_dir, 'gtInstances.json'), 'w') as f:
        content = json.dump(new_content, f)


if __name__ == '__main__':
    print(
'''======================= NOTE =======================
To use this script, the name of your instance index 
file should EXACTLY follow the scheme:

    <city_name>_<id>_<id>_leftImg8bit_pred.txt

e.g.

    frankfurt_000001_042098_leftImg8bit_pred.txt

. Otherwise the script may not function correctly.
====================================================
'''
)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--offset')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument(
        '--dataset-dir', default='/msravcshare/dataset/original_cityscapes/')
    parser.add_argument('--out')
    args = parser.parse_args()

    in_dir = args.input
    if args.offset is None:
        if args.split == 'val':
            offset_dir = osp.join(DATA_ROOT, 'cityscapes', 'val', 'offset_pred', 'instance', 'offset_hrnext')
        else:
            offset_dir = osp.join(DATA_ROOT, 'cityscapes', 'test_offset', 'instance', 'offset_hrnext')
    else:
        offset_dir = args.offset
    if args.out is not None:
        out_dir = args.out
    else:
        out_dir = osp.join(in_dir, 'label_w_segfix')

    os.makedirs(out_dir, exist_ok=True)
    input_args = [fn for fn in os.listdir(in_dir) if fn.endswith('pred.txt')]
    print(len(input_args), 'files in total.')
    copy_gt()
    # mpp.Pool(processes=None).map(process, input_args)
    if args.split == 'val':
        evaluation(out_dir)
