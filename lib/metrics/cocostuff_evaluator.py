#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: RainbowSecret(yuyua@microsoft.com)


import argparse
import os
import pdb

import numpy as np

from lib.utils.helpers.image_helper import ImageHelper
from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.configer import Configer
from lib.metrics.running_score import RunningScore


class COCOStuffEvaluator(object):
    def __init__(self, configer):
        self.configer = configer
        self.seg_running_score = RunningScore(configer)
        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 
                              11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 
                              21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 
                              33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 
                              42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 
                              52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 
                              61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 
                              74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 
                              84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80, 92: 81, 93: 82, 
                              94: 83, 95: 84, 96: 85, 97: 86, 98: 87, 99: 88, 100: 89, 101: 90, 102: 91, 
                              103: 92, 104: 93, 105: 94, 106: 95, 107: 96, 108: 97, 109: 98, 110: 99, 
                              111: 100, 112: 101, 113: 102, 114: 103, 115: 104, 116: 105, 117: 106, 118: 107, 
                              119: 108, 120: 109, 121: 110, 122: 111, 123: 112, 124: 113, 125: 114, 126: 115, 
                              127: 116, 128: 117, 129: 118, 130: 119, 131: 120, 132: 121, 133: 122, 134: 123, 
                              135: 124, 136: 125, 137: 126, 138: 127, 139: 128, 140: 129, 141: 130, 142: 131, 
                              143: 132, 144: 133, 145: 134, 146: 135, 147: 136, 148: 137, 149: 138, 150: 139, 
                              151: 140, 152: 141, 153: 142, 154: 143, 155: 144, 156: 145, 157: 146, 158: 147, 
                              159: 148, 160: 149, 161: 150, 162: 151, 163: 152, 164: 153, 165: 154, 166: 155, 
                              167: 156, 168: 157, 169: 158, 170: 159, 171: 160, 172: 161, 173: 162, 174: 163, 
                              175: 164, 176: 165, 177: 166, 178: 167, 179: 168, 180: 169, 181: 170, 182: 171,
                              12: 0, 26: 0, 29: 0, 30: 0, 45: 0, 66: 0, 68: 0, 69: 0, 71: 0, 83: 0, 91: 0}


    def relabel(self, labelmap):
        # label
        label_copy = labelmap.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[labelmap == k] = v
        return label_copy.astype(np.uint8)

    def reduce_one(self, labelmap):
        return (labelmap - 1).astype(np.uint8)

    def add_one(self, labelmap):
        return (labelmap + 1).astype(np.uint8)

    def evaluate(self, pred_dir, gt_dir):
        img_cnt = 0
        for filename in os.listdir(pred_dir):
            print(filename)
            pred_path = os.path.join(pred_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            predmap = ImageHelper.img2np(ImageHelper.read_image(pred_path, tool='pil', mode='P'))
            gtmap = ImageHelper.img2np(ImageHelper.read_image(gt_path, tool='pil', mode='P'))

            predmap = self.relabel(predmap)
            gtmap = self.relabel(gtmap)
            gtmap[gtmap == 0] = 255

            self.seg_running_score.update(predmap[np.newaxis, :, :], gtmap[np.newaxis, :, :])
            img_cnt += 1

        Log.info('Evaluate {} images'.format(img_cnt))
        Log.info('mIOU: {}'.format(self.seg_running_score.get_mean_iou()))
        Log.info('Pixel ACC: {}'.format(self.seg_running_score.get_pixel_acc()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The configs file of pose.')
    parser.add_argument('--gt_dir', default=None, type=str,
                        dest='gt_dir', help='The groundtruth annotations.')
    parser.add_argument('--pred_dir', default=None, type=str,
                        dest='pred_dir', help='The label dir of predict annotations.')
    args = parser.parse_args()

    cocostuff_evaluator = COCOStuffEvaluator(Configer(configs=args.configs))
    cocostuff_evaluator.evaluate(args.pred_dir, args.gt_dir)
