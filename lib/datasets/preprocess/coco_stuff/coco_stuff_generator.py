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
    def __init__(self):
        with (args.ori_root_dir / 'cocostuff-10k-v1.1.json').open() as fd:
            data = json.load(fd)

        images = {}
        n2id = {}
        for image_meta in data['images']:
            images[image_meta['id']] = {
                'file_name': image_meta['file_name'],
                'hw': (image_meta['height'], image_meta['width']),
                'segm': []
            }
            n2id[image_meta['file_name'].rpartition('.')[0]] = image_meta['id']

        for an_meta in data['annotations']:
            image_id = an_meta['image_id']
            pg = an_meta['segmentation']
            images[image_id]['segm'].append([pg, an_meta['category_id']])

        self.images = images
        self.n2id = n2id

    def __getitem__(self, name):
        return self.images[self.n2id[name]]

    def build(self, name):
        image_meta = self[name]
        labelmap = np.zeros(image_meta['hw'], dtype=np.uint8)

        in_label = args.ori_root_dir / 'annotations' / (name + '.mat')
        mat = io.loadmat(str(in_label))
        aid2cid = mat['regionLabelsStuff']
        aid_map = mat['regionMapStuff']
        for aid in np.unique(aid_map):
            cid = aid2cid[aid - 1]
            if cid == 0:
                continue
            labelmap[aid_map == aid] = cid

        for segm, cid in image_meta['segm']:
            if isinstance(segm, list):
                if len(segm[0]) < 6 or len(segm[0]) % 2 != 0:
                    continue

                frpyobj = mask_util.frPyObjects(
                    segm,
                    *image_meta['hw'],
                )
            elif isinstance(segm, dict):
                if isinstance(segm["counts"], list):
                    # convert to compressed RLE
                    frpyobj = [mask_util.frPyObjects(segm, *segm["size"])]

            mask = mask_util.decode(mask_util.merge(frpyobj)).astype(np.bool)
            labelmap[mask] = cid

        return labelmap


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
        validate_labelmap = np.array(Image.open(str(validate_label))).astype(np.uint8)
        diff = (validate_labelmap != labelmap).sum() / labelmap.size * 100
        if diff > 1:
            print('{:.6f}%'.format(diff))
        equal = (np.unique(validate_labelmap) == np.unique(labelmap))
        # assert equal if isinstance(equal, bool) else equal.all()

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
    # for arg in input_args():
    #     process(arg)