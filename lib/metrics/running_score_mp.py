##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: JingyiXie, RainbowSecret, Donny You
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np

class RunningScore(object):

    def __init__(self, configer, num_classes=None, ignore_index=None):
        self.configer = configer
        if num_classes is None:
            self.n_classes = self.configer.get('data', 'num_classes')
        else:
            self.n_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def get_F1_score(self):
        assert self.n_classes == 2
        TN, FN, FP, TP = self.confusion_matrix.flatten()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return 2 / (1 / precision + 1 / recall), precision, recall        

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        if self.ignore_index is not None:
            mask = mask & (label_true != self.ignore_index)
            
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2)

        # print(np.unique(label_true))
        # print(np.unique(label_pred))
        hist = hist.reshape(n_class, n_class)

        return hist

    def hist(self, label_preds, label_trues):
        cm = 0.
        for lt, lp in zip(label_trues, label_preds):
            cm = cm + self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        return cm

    def gather_hist(self, hists):
        for x in hists:
            self.confusion_matrix += x

    def _get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        # print('category-wise mean accuracy: ', acc_cls)

        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # print('category-wise mean iou: ', iu)

        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return acc, acc_cls, fwavacc, mean_iu, cls_iu

    def get_mean_iou(self):
        return self._get_scores()[3]

    def get_cls_iu(self):
        return self._get_scores()[4]

    def get_pixel_acc(self):
        return self._get_scores()[0]

    def get_mean_acc(self):
        return self._get_scores()[1]

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))