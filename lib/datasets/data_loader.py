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

import torch
from torch.utils import data

import lib.datasets.tools.transforms as trans
import lib.datasets.tools.cv2_aug_transforms as cv2_aug_trans
import lib.datasets.tools.pil_aug_transforms as pil_aug_trans
from lib.datasets.loader.default_loader import DefaultLoader, CSDataTestLoader
from lib.datasets.loader.ade20k_loader import ADE20KLoader
from lib.datasets.loader.lip_loader import LipLoader
from lib.datasets.tools.collate import collate
from lib.utils.tools.logger import Logger as Log

from lib.utils.distributed import get_world_size, get_rank, is_distributed

import pdb


class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
            self.aug_val_transform = pil_aug_trans.PILAugCompose(self.configer, split='val')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='val')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])

        self.label_transform = trans.Compose([
            trans.ToLabel(),
            trans.ReLabel(255, -1), ])


    def get_trainloader(self):
        if self.configer.exists('data', 'use_edge') and self.configer.get('data', 'use_edge') == 'ce2p':
            """
            ce2p manner:
            load both the ground-truth label and edge.
            """
            Log.info('use edge (follow ce2p) for train...')
            trainloader = data.DataLoader(
                LipLoader(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                          aug_transform=self.aug_train_transform,
                          img_transform=self.img_transform,
                          label_transform=self.label_transform,
                          configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'),
                shuffle=True, drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )
            return trainloader

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
            trainloader = data.DataLoader(
                ADE20KLoader(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                             aug_transform=self.aug_train_transform,
                             img_transform=self.img_transform,
                             label_transform=self.label_transform,
                             configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'),
                shuffle=True, drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )
            return trainloader

        else:
            """
            Default manner:
            support input images of the same shapes.
            """
            dataset = DefaultLoader(
                root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                aug_transform=self.aug_train_transform,
                img_transform=self.img_transform,
                label_transform=self.label_transform,
                configer=self.configer
            )
            if is_distributed():
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)            
            else:
                sampler = None
            Log.info('use the DefaultLoader for train...')
            trainloader = data.DataLoader(
                dataset,
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

        if self.configer.get('method') == 'fcn_segmentor':
            """
            default manner:
            load the ground-truth label.
            """   
            Log.info('use DefaultLoader for val ...')
            valloader = data.DataLoader(
                DefaultLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                              aug_transform=self.aug_val_transform,
                              img_transform=self.img_transform,
                              label_transform=self.label_transform,
                              configer=self.configer),
                batch_size=self.configer.get('val', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'), shuffle=False,
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('val', 'data_transformer')
                )
            )
            return valloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None


    def get_testloader(self, dataset=None):
            dataset = 'test' if dataset is None else dataset

            if self.configer.get('method') == 'fcn_segmentor':
                Log.info('use CSDataTestLoader for test ...')
                testloader = data.DataLoader(
                    CSDataTestLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                                     img_transform=self.img_transform,
                                     configer=self.configer),
                    batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
                    num_workers=self.configer.get('data', 'workers'), shuffle=False,
                    collate_fn=lambda *args: collate(
                        *args, trans_dict=self.configer.get('test', 'data_transformer')
                    )
                )
                return testloader


if __name__ == "__main__":
    pass
