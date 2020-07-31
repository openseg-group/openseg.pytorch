import os
import cv2
import sys
import torch
import argparse
import subprocess
import numpy as np
from PIL import Image
from glob import glob
import os.path as osp
import scipy.io as io
import multiprocessing as mp
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import multiprocessing.pool as mpp
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt

script_path = osp.abspath(osp.join(osp.dirname(__file__)))
os.chdir(osp.join(script_path, '..', '..', '..', '..'))
sys.path.insert(0, os.getcwd())
os.environ['PYTHONPATH'] = os.getcwd() + ':' + os.environ.get('PYTHONPATH', '')

DATA_ROOT = subprocess.check_output(
    ['bash', '-c', "source config.profile; echo $DATA_ROOT"]
).decode().strip()


def sobel_kernel(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    k = np.zeros(shape)
    p = [
        (j, i)
        for j in range(shape[0])
        for i in range(shape[1])
        if not (i == (shape[1] - 1) / 2.0 and j == (shape[0] - 1) / 2.0)
    ]

    for j, i in p:
        j_ = int(j - (shape[0] - 1) / 2.0)
        i_ = int(i - (shape[1] - 1) / 2.0)
        k[j, i] = (i_ if axis == 0 else j_) / float(i_ * i_ + j_ * j_)
    return torch.from_numpy(k).unsqueeze(0)

label_list = set([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
stuff_label_list = set(range(24))
inst_label_list = label_list - stuff_label_list

from lib.utils.helpers.offset_helper import DTOffsetHelper

def _vis_offset(_offset, image_name, out_dir, image=None, color=(0, 0, 255)):
    if image is None:
        color = 255
        image = np.zeros_like(_offset[:, :, 0], dtype=np.uint8)

    step = 3
    coord_map = torch.stack(torch.meshgrid([torch.arange(
        length) for length in _offset.shape[:-1]]), dim=-1).numpy().astype(np.int)
    offset = (_offset * 5 + coord_map).astype(np.int)
    for i in range(step//2, offset.shape[0], step):
        for j in range(step//2, offset.shape[1], step):
            if (_offset[i, j] == 0).all():
                continue
            cv2.arrowedLine(img=image, pt1=tuple(
                coord_map[i, j][::-1]), pt2=tuple(offset[i, j][::-1]), color=color, thickness=1)
    cv2.imwrite(os.path.join(out_dir, '{}.png'.format(image_name)), image)

def process(inp):
    (indir, outdir, basename) = inp
    print(inp)
    labelmap = np.array(Image.open(osp.join(indir, basename)))
    depth_map = np.ones(labelmap.shape) * 0
    dir_map = np.zeros((*labelmap.shape, 2))

    ignore_id_list = set(range(256)) - label_list

    for id in ignore_id_list:
        labelmap[labelmap == id] = 255
    
    labelmap_flattened = np.unique(labelmap)
    print(labelmap_flattened)

    for id in labelmap_flattened:
        labelmap_i = labelmap.copy()
        labelmap_i[labelmap != id] = 0
        labelmap_i[labelmap == id] = 1

        # if labelmap_i.sum() < 100:
        #     continue

        if args.metric == 'euc':
            depth_i = distance_transform_edt(labelmap_i)
        elif args.metric == 'taxicab':
            depth_i = distance_transform_cdt(labelmap_i, metric='taxicab')
        else:
            raise RuntimeError
        depth_map[labelmap_i == 1] = depth_i[labelmap_i == 1]

        dir_i_before = dir_i = np.zeros_like(dir_map)
        dir_i = torch.nn.functional.conv2d(torch.from_numpy(depth_i).float().view(1, 1, *depth_i.shape), sobel_ker, padding=ksize//2).squeeze().permute(1, 2, 0).numpy()

        # The following line is necessary
        dir_i[(labelmap_i == 0), :] = 0
        
        dir_map += dir_i
    depth_map[depth_map > 250] = 250
    depth_map = depth_map.astype(np.uint8)
    # print(np.unique(depth_map))
    deg_reduce = 2
    dir_deg_map = np.degrees(np.arctan2(dir_map[:, :, 0], dir_map[:, :, 1])) + 180
    dir_deg_map = (dir_deg_map / deg_reduce)
    print(dir_deg_map.min(), dir_deg_map.max())
    dir_deg_map = dir_deg_map.astype(np.uint8) 
    dct = {"dir_deg": dir_deg_map, "depth": depth_map, 'deg_reduce': deg_reduce}
    safe_savemat(
        osp.join(outdir, basename.replace("png", "mat")),
        dct
    )


def safe_savemat(fn, dct):
    io.savemat(fn, dct, do_compression=True)
    try:
        io.loadmat(fn)
    except Exception as e:
        print(e)
        io.savemat(fn, dct, do_compression=False)

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", dest='datadir', default=osp.join(DATA_ROOT,'cityscapes'))
parser.add_argument("--outname", default='offset_gt/dt_offset_inst_w_stuff')
parser.add_argument('--split', nargs='+', default=['val', 'train'])
parser.add_argument("--ksize", type=int, default=5)
parser.add_argument('--metric', default='euc', choices=['euc', 'taxicab'])
args = parser.parse_args()

ksize = args.ksize

sobel_x, sobel_y = (sobel_kernel((ksize, ksize), i) for i in (0, 1))
sobel_ker = torch.cat([sobel_y, sobel_x], dim=0).view(2, 1, ksize, ksize).float()

for dataset in args.split:
    indir = osp.join(args.datadir, dataset, 'instance')
    outdir = osp.join(args.datadir, dataset, args.outname)
    os.makedirs(outdir, exist_ok=True)
    args_to_apply = [(indir, outdir, osp.basename(basename)) for basename in glob(osp.join(indir, "*.png"))]
    mpp.Pool(processes=mp.cpu_count() // 2).map(process, args_to_apply)

