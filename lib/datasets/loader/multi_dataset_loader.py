# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: JingyiXie
# Microsoft Research
# hsfzxjy@gmail.com
# Copyright (c) 2020
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb

import numpy as np
import torch
from torch.utils import data
from lib.utils.tools.configer import Configer
from lib.datasets.tools.cv2_aug_transforms import CV2AugCompose

class MultiDatasetLoader(data.Dataset):
    """
    A meta dataloader that can serve data from multiple datasets.

    `root_dirs` is list of strings representing root directory of each dataset.
    `base_class` is an task-specific dataloader class, such as `DefaultLoader`, `OffsetLoader`.

    During training, this object will serve `N * MAX` items within an epoch, where `N` is 
    number of datasets, and `MAX` is maximum items number among the `N` datasets.

    Items with index `N * j + i` is guaranteed to be the j-th item from the i-th dataset,
    for j = 0..MAX-1, i=0..N-1. For dataset with length less than `MAX`, we will repeat its 
    items to get a list of length `MAX`.

    During training, this object should be used with `MultiDatasetTrainingSampler` to get a
    balance sampling among the `N` datasets.

    During validation, this object serve a list of items that is the concatenation of items from
    all datasets.
    """
    def __init__(self, root_dirs, base_class, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.configer = configer
        self.base_class = base_class
        self.dataset = dataset
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.child_loaders = self._get_child_loaders(root_dirs, base_class)
        self.num_datasets = len(self.child_loaders)
        self.selected_dataset_index = -1

    def _get_child_configer_transform(self, child_config_file):

        dataset_configer = Configer(configs=child_config_file)
        child_configer = self.configer.clone()

        child_configer.params_root['data'].update(dataset_configer.get('data'))

        if self.configer.exists('use_adaptive_transform') or self.dataset == 'val':
            child_configer.params_root.update({
                'train_trans': dataset_configer.params_root['train_trans'],
                'val_trans': dataset_configer.params_root['val_trans'],
            })

        return child_configer, CV2AugCompose(split=self.dataset, configer=child_configer)

    def _get_child_loaders(self, root_dirs, base_class):
        child_config_files = self.configer.get('child_config_files')
        child_loaders = []
        for i, root_dir in enumerate(root_dirs):
            child_configer, child_aug_transform = self._get_child_configer_transform(
                child_config_files[i]
            )
            print(child_aug_transform)
            child_loaders.append(
                base_class(
                    root_dir, child_aug_transform, self.dataset,
                    self.img_transform, self.label_transform, 
                    child_configer
                )
            )
        return child_loaders

    def __len__(self):
        if self.dataset == 'train':
            return self.num_datasets * max(len(loader) for loader in self.child_loaders)
        elif self.dataset == 'val':
            if self.selected_dataset_index >= 0:
                return len(self.child_loaders[self.selected_dataset_index])
            return sum(len(loader) for loader in self.child_loaders)

    def __getitem__(self, idx):

        if self.dataset == 'train':
            loader = self.child_loaders[idx % self.num_datasets]
            return loader[(idx // self.num_datasets) % len(loader)]

        elif self.dataset == 'val':

            if self.selected_dataset_index >= 0:
                return self.child_loaders[self.selected_dataset_index][idx]

            current_loader = None
            for loader in self.child_loaders:
                if idx < len(loader):
                    current_loader = loader
                    break
                idx -= len(loader)
            return current_loader[idx]

    def select(self, dataset_idx):
        assert 0 <= dataset_idx < self.num_datasets
        self.selected_dataset_index = dataset_idx


class MultiDatasetTrainingSampler(torch.utils.data.Sampler):

    def __init__(self, data_source):
        assert isinstance(data_source, MultiDatasetLoader)
        assert data_source.dataset == 'train'
        self.data_source = data_source

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source) // self.data_source.num_datasets
        return get_multi_randperm(n, self.data_source.num_datasets)

    def __len__(self):
        return self.num_samples


def get_multi_randperm(n, m):
    """
    Return an iterator of length n * m.

    Say x_i is the i-th element yielded, i = 0...n * m - 1.
    x_i will always have the same remainder as i, modulo m.
    Fix i, {x_{m * j + i}} forms a permutation of [0..n-1].
    """
    for idx_group in zip(*[torch.randperm(n).tolist() for _ in range(m)]):
        for loader_id, idx in enumerate(idx_group):
            yield idx * m + loader_id


if __name__ == '__main__':
    print(list(get_multi_randperm(4, 3)))
