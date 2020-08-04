import os
import json
import shutil
import argparse
from pathlib import Path
import multiprocessing as mp
import multiprocessing.pool as mpp
from collections import defaultdict

import numpy as np
import scipy.io as io
from PIL import Image
import pycocotools.mask as mask_util


class COCOProcessor:
    def build(self, name):
        in_label = args.ori_root_dir / 'annotations' / (name + '.mat')
        return io.loadmat(str(in_label))['S'].astype(np.uint8)


def process(inputs):
    split, name = inputs
    print('Processing', name, split)
    in_img = args.ori_root_dir / 'images' / (name + '.jpg')
    out_img: Path = args.save_dir / split / 'images' / (name + '.jpg')
    out_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(in_img), str(out_img))

    out_label: Path = args.save_dir / split / 'label' / (name + '.png')
    labelmap = coco.build(name)

    if args.validate_dir is not None:
        validate_label = args.validate_dir / split / 'label' / (name + '.png')
        validate_labelmap = np.array(Image.open(str(validate_label))).astype(
            np.uint8)
        diff = (validate_labelmap != labelmap).sum() / labelmap.size * 100
        if diff > 1:
            print('{:.6f}%'.format(diff))
        equal = (np.unique(validate_labelmap) == np.unique(labelmap))
        assert equal if isinstance(equal, bool) else equal.all()

    out_label.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labelmap).save(str(out_label))


def input_args():
    with (args.ori_root_dir / 'imageLists' / 'test.txt').open() as f:
        for name in f:
            yield ('val', name.strip())

    with (args.ori_root_dir / 'imageLists' / 'train.txt').open() as f:
        for name in f:
            yield ('train', name.strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_root_dir', type=Path)
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--validate_dir', type=lambda x: x and Path(x))
    args = parser.parse_args()

    coco = COCOProcessor()
    mpp.Pool(processes=None).map(process, input_args())
