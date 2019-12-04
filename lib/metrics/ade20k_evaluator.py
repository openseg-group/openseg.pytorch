#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)

import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.abspath(osp.dirname(__file__)), '..', '..'))

import argparse
import os
import pdb

import numpy as np

from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.configer import Configer
from lib.metrics.running_score_mp import RunningScore


class ADE20KEvaluator(object):
    def __init__(self, configer):
        self.configer = configer
        self.seg_running_score = RunningScore(configer)

    def relabel(self, labelmap):
        return (labelmap - 1).astype(np.uint8)

    def _encode_label(self, labelmap):

        if not self.configer.exists('data', 'label_list'):
            return labelmap

        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.int) * 255
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def _mp_target(self, inp):
        filename, pred_dir, gt_dir = inp
        print(filename)
        
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        try:
            predmap = self._encode_label(ImageHelper.img2np(ImageHelper.read_image(pred_path, tool='pil', mode='P')))
            gtmap = self._encode_label(ImageHelper.img2np(ImageHelper.read_image(gt_path, tool='pil', mode='P')))
        except Exception as e:
            print(e)
            return 0.

        if "pascal_context" in gt_dir or "ADE" in gt_dir:
            predmap = self.relabel(predmap)
            gtmap = self.relabel(gtmap)

        return self.seg_running_score.hist(predmap[np.newaxis, :, :], gtmap[np.newaxis, :, :])

    def evaluate(self, pred_dir, gt_dir):
        # img_cnt = 0
        import multiprocessing.pool as mpp

        file_list = args.file_list
        if file_list != 'all':
            with open(osp.join(args.gt_dir, '..', 'file_list', file_list)) as f:
                flist = set([x.strip() for x in f])
            input_args = [
                (filename, pred_dir, gt_dir)
                for filename in os.listdir(pred_dir) if filename in flist
            ]
        else:
            input_args = [
                (filename, pred_dir, gt_dir)
                for filename in os.listdir(pred_dir)
            ]

        Log.info('{} files in total.'.format(len(input_args)))
        hists = mpp.Pool().map(
            self._mp_target, 
            input_args
        )
        self.seg_running_score.gather_hist(hists)
        Log.info('Evaluate {} images'.format(len(hists)))
        Log.info('mIOU: {}'.format(self.seg_running_score.get_mean_iou()))
        Log.info('Pixel ACC: {}'.format(self.seg_running_score.get_pixel_acc()))
        Log.info('mIoU (Class-wise)')
        iou_dict = self.seg_running_score.get_cls_iu()
        for cid, miou in iou_dict.items():
            Log.info('\t{}\t{}'.format(cid, miou))
        print(' & '.join('{:.1f}'.format(x * 100) for x in list(iou_dict.values()) + [self.seg_running_score.get_mean_iou()]))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The configs file of pose.')
    parser.add_argument('--gt_dir', default=None, type=str,
                        dest='gt_dir', help='The groundtruth annotations.')
    parser.add_argument('--pred_dir', default=None, type=str,
                        dest='pred_dir', help='The label dir of predict annotations.')
    parser.add_argument('--file_list', default='all')
    args = parser.parse_args()

    ade20k_evaluator = ADE20KEvaluator(Configer(configs=args.configs))
    ade20k_evaluator.evaluate(args.pred_dir, args.gt_dir)