#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

ORI_ROOT_DIR='/msravcshare/dataset/lip/humanparsing'
SAVE_DIR='/msravcshare/dataset/lip/atr'

mkdir -p ${SAVE_DIR}

#directly copy images
mkdir -p ${SAVE_DIR}/train
cp -r ${ORI_ROOT_DIR}/JPEGImages ${SAVE_DIR}/train/image

${PYTHON} -u atr_generator.py --ori_root_dir $ORI_ROOT_DIR \
                          --save_dir $SAVE_DIR
