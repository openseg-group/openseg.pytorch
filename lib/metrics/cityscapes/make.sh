#!/usr/bin/env bash

# check the enviroment info

PYTHON="/root/miniconda3/bin/python"
export PYTHONPATH="/msravcshare/yuyua/code/segmentation/openseg.pytorch":$PYTHONPATH

cd ../../../
${PYTHON} lib/metrics/cityscapes/setup.py build_ext --inplace
