#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
# PYTHON="/root/miniconda3/bin/python"
PYTHON="/data/anaconda/envs/py35/bin/python"

ORI_ROOT_DIR='/msravcshare/dataset/mapillary-vista-v1.1'
SAVE_DIR='/msravcshare/dataset/cityscapes/mapillary'

mkdir -p ${SAVE_DIR}

# directly copy images
# mkdir -p ${SAVE_DIR}/train
# cp -r ${ORI_ROOT_DIR}/training/images ${SAVE_DIR}/train/image

# mkdir -p ${SAVE_DIR}/val
# cp -r ${ORI_ROOT_DIR}/validation/images ${SAVE_DIR}/val/image


${PYTHON} mapillary_generator.py --ori_root_dir $ORI_ROOT_DIR \
                          --save_dir $SAVE_DIR