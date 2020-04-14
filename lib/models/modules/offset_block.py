import torch
import math
import pdb
from torch import nn
from torch.nn import functional as F
import numpy as np

from lib.models.tools.module_helper import ModuleHelper


class OffsetBlock(nn.Module):
    '''
    This module takes relative offset as input and outputs feature at each position (coordinate + offset)
    '''
    def __init__(self):
        super(OffsetBlock, self).__init__()
        self.coord_map = None
        self.norm_factor = None
    
    def _gen_coord_map(self, H, W):
        coord_vecs = [torch.arange(length, dtype=torch.float).cuda() for length in (H, W)]
        coord_h, coord_w = torch.meshgrid(coord_vecs)
        return coord_h, coord_w
    
    def forward(self, x, offset_map):
        n, c, h, w = x.size()
        
        if self.coord_map is None or self.coord_map[0].size() != offset_map.size()[2:]:
            self.coord_map = self._gen_coord_map(h, w)
            self.norm_factor = torch.cuda.FloatTensor([(w-1) / 2, (h-1) / 2])
        
        # offset to absolute coordinate
        grid_h = offset_map[:, 0] + self.coord_map[0]                               # (N, H, W)
        grid_w = offset_map[:, 1] + self.coord_map[1]                               # (N, H, W)

        # scale to [-1, 1], order of grid: [x, y] (i.e., [w, h])
        grid = torch.stack([grid_w, grid_h], dim=-1) / self.norm_factor - 1.        # (N, H, W, 2)

        # use grid to obtain output feature
        feats = F.grid_sample(x, grid, padding_mode='border')                       # (N, C, H, W)
        
        return feats


class OffsetModule(nn.Module):
    def __init__(self):
        super(OffsetModule, self).__init__()
        self.offset_block = OffsetBlock()
    
    def forward(self, x, offset):
        # sample
        x_out = self.offset_block(x, offset)
        return x_out
