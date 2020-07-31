##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret, JingyiXie
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
import torch
from torch.utils import data

import lib.datasets.tools.transforms as trans
import lib.datasets.tools.cv2_aug_transforms as cv2_aug_trans
import lib.datasets.tools.pil_aug_transforms as pil_aug_trans
from lib.datasets.loader.default_loader import DefaultLoader, CSDataTestLoader
from lib.datasets.loader.ade20k_loader import ADE20KLoader
from lib.datasets.loader.lip_loader import LipLoader
from lib.datasets.loader.offset_loader import DTOffsetLoader
from lib.datasets.tools.collate import collate
from lib.utils.tools.logger import Logger as Log

from lib.utils.distributed import get_world_size, get_rank, is_distributed


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        from lib.datasets.tools import cv2_aug_transforms
        self.aug_train_transform = cv2_aug_transforms.CV2AugCompose(self.configer, split='train')
        self.aug_val_transform = cv2_aug_transforms.CV2AugCompose(self.configer, split='val')

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])

        self.label_transform = trans.Compose([
            trans.ToLabel(),
            trans.ReLabel(255, -1), ])

    def get_dataloader_sampler(self, klass, split, dataset):

        from lib.datasets.loader.multi_dataset_loader import MultiDatasetLoader, MultiDatasetTrainingSampler

        root_dir = self.configer.get('data', 'data_dir')
        if isinstance(root_dir, list) and len(root_dir) == 1:
            root_dir = root_dir[0]

        kwargs = dict(
            dataset=dataset,
            aug_transform=(self.aug_train_transform if split == 'train' else self.aug_val_transform),
            img_transform=self.img_transform,
            label_transform=self.label_transform,
            configer=self.configer
        )

        if isinstance(root_dir, str):
            loader = klass(root_dir, **kwargs)
            multi_dataset = False
        elif isinstance(root_dir, list):
            loader = MultiDatasetLoader(root_dir, klass, **kwargs)
            multi_dataset = True
            Log.info('use multi-dataset for {}...'.format(dataset))
        else:
            raise RuntimeError('Unknown root dir {}'.format(root_dir))

        if split == 'train':
            if is_distributed() and multi_dataset:
                raise RuntimeError('Currently multi dataset doesn\'t support distributed.')

            if is_distributed():
                sampler = torch.utils.data.distributed.DistributedSampler(loader)
            elif multi_dataset:
                sampler = MultiDatasetTrainingSampler(loader)
            else:
                sampler = None

        elif split == 'val':

            if is_distributed():
                sampler = torch.utils.data.distributed.DistributedSampler(loader)
            else:
                sampler = None

        return loader, sampler

    def get_trainloader(self):
        if self.configer.exists('data', 'use_edge') and self.configer.get('data', 'use_edge') == 'ce2p':
            """
            ce2p manner:
            load both the ground-truth label and edge.
            """
            Log.info('use edge (follow ce2p) for train...')
            klass = LipLoader

        elif self.configer.exists('data', 'use_dt_offset') or self.configer.exists('data', 'pred_dt_offset'):
            """
            dt-offset manner:
            load both the ground-truth label and offset (based on distance transform).
            """
            Log.info('use distance transform offset loader for train...')
            klass = DTOffsetLoader

        elif self.configer.exists('train', 'loader') and \
            (self.configer.get('train', 'loader') == 'ade20k' 
             or self.configer.get('train', 'loader') == 'pascal_context'
             or self.configer.get('train', 'loader') == 'pascal_voc'
             or self.configer.get('train', 'loader') == 'coco_stuff'):
            """
            ADE20KLoader manner:
            support input images of different shapes.
            """
            Log.info('use ADE20KLoader (diverse input shape) for train...')
            klass = ADE20KLoader
        else:
            """
            Default manner:
            + support input images of the same shapes.
            + support distributed training (the performance is more un-stable than non-distributed manner)
            """
            Log.info('use the DefaultLoader for train...')
            klass = DefaultLoader
        loader, sampler = self.get_dataloader_sampler(klass, 'train', 'train')
        trainloader = data.DataLoader(
            loader,
            batch_size=self.configer.get('train', 'batch_size') // get_world_size(), pin_memory=True,
            num_workers=self.configer.get('data', 'workers') // get_world_size(),
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=self.configer.get('data', 'drop_last'),
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('train', 'data_transformer')
            )
        )
        return trainloader
            

    def get_valloader(self, dataset=None):
        dataset = 'val' if dataset is None else dataset

        if self.configer.exists('data', 'use_dt_offset') or self.configer.exists('data', 'pred_dt_offset'):
            """
            dt-offset manner:
            load both the ground-truth label and offset (based on distance transform).
            """   
            Log.info('use distance transform based offset loader for val ...')
            klass = DTOffsetLoader

        elif self.configer.get('method') == 'fcn_segmentor':
            """
            default manner:
            load the ground-truth label.
            """   
            Log.info('use DefaultLoader for val ...')
            klass = DefaultLoader
        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

        loader, sampler = self.get_dataloader_sampler(klass, 'val', dataset)
        valloader = data.DataLoader(
            loader,
            sampler=sampler,
            batch_size=self.configer.get('val', 'batch_size') // get_world_size(), pin_memory=True,
            num_workers=self.configer.get('data', 'workers'), shuffle=False,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('val', 'data_transformer')
            )
        )
        return valloader

    def get_testloader(self, dataset=None):
            dataset = 'test' if dataset is None else dataset
            if self.configer.exists('data', 'use_sw_offset') or self.configer.exists('data', 'pred_sw_offset'):
                Log.info('use sliding window based offset loader for test ...')
                test_loader = data.DataLoader(
                    SWOffsetTestLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                                       img_transform=self.img_transform,
                                       configer=self.configer),
                    batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
                    num_workers=self.configer.get('data', 'workers'), shuffle=False,
                    collate_fn=lambda *args: collate(
                        *args, trans_dict=self.configer.get('test', 'data_transformer')
                    )
                )
                return test_loader

            elif self.configer.get('method') == 'fcn_segmentor':
                Log.info('use CSDataTestLoader for test ...')
                test_loader = data.DataLoader(
                    CSDataTestLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                                     img_transform=self.img_transform,
                                     configer=self.configer),
                    batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
                    num_workers=self.configer.get('data', 'workers'), shuffle=False,
                    collate_fn=lambda *args: collate(
                        *args, trans_dict=self.configer.get('test', 'data_transformer')
                    )
                )
                return test_loader


if __name__ == "__main__":
    pass
