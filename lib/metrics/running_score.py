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
import lib.utils.distributed as dist


class SimpleCounterRunningScore(object):

    def __init__(self, ignore_index=-1):
        self.correct_count = 0
        self.total_count = 0
        self.reduced_counters = None

    def update(self, pred, gt):
        self.reduced_counters = None
        self.correct_count += (pred == gt).sum()
        self.total_count += (gt != -1).sum()

    def reduce_scores(self):
        counters = np.array([self.total_count, self.correct_count])
        if dist.is_distributed():
            counters = dist.all_reduce_numpy(counters)
        self.reduced_counters = counters

    def get_mean_acc(self):
        if self.reduced_counters is None:
            self.reduce_scores()
        total_count, correct_count = list(self.reduced_counters)        
        return correct_count / max(1, total_count)

    def reset(self):
        self.correct_count = self.total_count = 0
        self.reduced_counters = None
        

class MultiLabelRunningScore(object):
    """
    Suppose label[p] is N-dim multi-hot vector, and pred[p] is N-dim logits. THRESHOLD is the threshold.
    We consider a location `p` as correct, if either is true:
     1) label[p] has at least one non-zero elements and label[p][argmax(pred[p])] == 1, i.e., prediction with highest confidence is correct.
     2) label[p] are all zeros, and all elements of pred[p] are lower than threshold.
    """

    def __init__(self, ignore_index=-1):
        self.ignore_index = ignore_index
        self.correct_count = 0
        self.total_count = 0
        self.reduced_counters = None

    def update(self, dir_pred, dir_gt, keep_mask):
        self.reduced_counters = None
        keep_mask = keep_mask & (dir_gt.sum(axis=-1) > 0)
        dir_gt = dir_gt[keep_mask, :]

        dir_pred = dir_pred[keep_mask]
        guess_index = dir_pred
        no_offset_mask = dir_pred == dir_gt.shape[-1]
        dir_pred[no_offset_mask] = 0

        guess_index = np.arange(guess_index.shape[0]) * dir_gt.shape[-1] + guess_index

        correct = np.take(
            dir_gt,
            guess_index,
        )

        correct = ((correct != 0) & ~no_offset_mask).sum()
        total = dir_gt.shape[0]

        self.total_count += total
        self.correct_count += correct

    def reduce_scores(self):
        counters = np.array([self.total_count, self.correct_count])
        if dist.is_distributed():
            counters = dist.all_reduce_numpy(counters)
        self.reduced_counters = counters

    def _get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        if self.reduced_counters is None:
            self.reduce_scores()
        total_count, correct_count = list(self.reduced_counters)
        return correct_count / max(1, total_count), 0, 0, 0

    def get_mean_iou(self):
        return self._get_scores()[3]

    def get_pixel_acc(self):
        return self._get_scores()[0]

    def get_mean_acc(self):
        return self._get_scores()[1]

    def reset(self):
        self.total_count = self.correct_count = 0
        self.reduced_counters = None



class RunningScore(object):

    def __init__(self, configer, num_classes=None, ignore_index=None):
        self.configer = configer
        if num_classes is None:
            self.n_classes = self.configer.get('data', 'num_classes')
        else:
            self.n_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.reduced_confusion_matrix = None

    def get_F1_score(self):
        assert self.n_classes == 2
        TN, FN, FP, TP = self.confusion_matrix.flatten()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return 2 / (1 / precision + 1 / recall), precision, recall       

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        mask &= (label_pred >= 0) & (label_pred < n_class)

        if self.ignore_index is not None:
            mask = mask & (label_true != self.ignore_index)
            
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2)

        # print(np.unique(label_true))
        # print(np.unique(label_pred))
        hist = hist.reshape(n_class, n_class)

        return hist

    def update(self, label_preds, label_trues):
        self.reduced_confusion_matrix = None
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def reduce_scores(self):
        if dist.is_distributed():
            hist = dist.all_reduce_numpy(self.confusion_matrix)
        else:
            hist = self.confusion_matrix
        self.reduced_confusion_matrix = hist

    def _get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        if self.reduced_confusion_matrix is None:
            self.reduce_scores()
        hist = self.reduced_confusion_matrix

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls_list = acc_cls = np.diag(hist) / hist.sum(axis=1)
        # print(acc_cls)
        # print('category-wise mean accuracy: ', acc_cls)

        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # print(iu)
        # print(np.nanmean(iu[iu > 0]))
        # print('category-wise mean iou: ', iu)

        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return acc, acc_cls_list, fwavacc, mean_iu, cls_iu

    def get_mean_iou(self):
        return self._get_scores()[3]

    def get_pixel_acc(self):
        return self._get_scores()[0]

    def get_mean_acc(self):
        return self._get_scores()[1]

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.reduced_confusion_matrix = None
