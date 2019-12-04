#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Lang Huang(layenhuang@outlook.com)
# Pascal context aug data generator.

PYTHON="/root/miniconda3/envs/pytorch1.0/bin/python"
ORI_ROOT_DIR='/msravcshare/dataset/pascal_context/' #'/msravcshare/dataset/pcontext/'
SAVE_DIR='/msravcshare/dataset/pascal_context/' #'/msravcshare/dataset/pcontext/'
SCRIPT_DIR='/msravcshare/yuyua/code/segmentation/openseg.pytorch/lib/datasets/preprocess/pascal_context'

cd ${ORI_ROOT_DIR}

# if [ ! -f train.pth ]; then
#     echo "Download training annotations"
#     wget https://hangzh.s3.amazonaws.com/encoding/data/pcontext/train.pth
# fi

# if [ ! -f val.pth ]; then
#     echo "Download val annotations"
#     wget https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth
# fi

cd ${SCRIPT_DIR}
echo "Start generation..."

python pascal_context_generator.py --ori_root_dir ${ORI_ROOT_DIR} \
                           --save_dir ${SAVE_DIR}

