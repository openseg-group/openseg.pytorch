#!/usr/bin/env bash

# check the enviroment info

# ss: 81.8

# ms v1: 82.8
# "scale_search": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
# "scale_weights": [1, 1, 2, 2, 2, 2]

# ms v2: 82.7
# "scale_search": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
# "scale_weights": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],

# ms v3: 82.8
# "scale_search": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
# "scale_weights": [1, 1, 2, 2, 2, 2, 2],

# ms v4: 82.7 -> 82.6
# "scale_search": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
# "scale_weights": [1, 1, 2, 2, 2, 2, 3]

# ms v5: 82.8
# "scale_search": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.25],
# "scale_weights": [1, 1, 2, 2, 2, 2, 1],

# ms v6: 82.6
# "scale_search": [0.5, 1.0, 2.0],
# "scale_weights": [1, 1, 1],

# with depth mIoU=82.8, pAcc=84.72

# [0.5, 1.0, 1.5, 2.0], mIoU=82.9


# with depth v3: base=0.5 [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  82.9
# with depth v3_p3: base=0.3 [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] 82.7
# with depth v3_p6: base=0.6 [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] 82.9
# with depth v3_p8: base=0.8 [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] 82.9

# with depth v4: base=0.8  [0.5, 1.0, 2.0]


nvidia-smi
PYTHON="/root/miniconda3/bin/python"
${PYTHON} -m pip install yacs
${PYTHON} -m pip install torchcontrib
${PYTHON} -m pip install pydensecrf

export PYTHONPATH="/msravcshare/yuyua/code/segmentation/openseg.pytorch":$PYTHONPATH

cd ../../../

DATA_DIR="/msravcshare/dataset/cityscapes"
SAVE_DIR="/msravcshare/dataset/seg_result/cityscapes/"
BACKBONE="hrnet48"
CONFIGS="configs/cityscapes/${BACKBONE}.json"
CONFIGS_TEST="configs/cityscapes/${BACKBONE}_test.json"
CONFIGS_TEST_DEPTH="configs/cityscapes/${BACKBONE}_test_depth.json"

MAX_ITERS=80000
BATCH_SIZE=8

MODEL_NAME="hrnet48_asp_ocr"
LOSS_TYPE="fs_auxohemce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_${BN_TYPE}_${BATCH_SIZE}_${MAX_ITERS}_OHEM_"$2
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"

PRETRAINED_MODEL="./pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --train_batch_size ${BATCH_SIZE} \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} \
                       > ${LOG_FILE} 2>&1
                       

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --train_batch_size ${BATCH_SIZE} \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --gpu 0 1 2 3 \
                       --resume_continue y --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME}  \
                        >> ${LOG_FILE} 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --phase debug --gpu 0 --log_to_file n  > ${LOG_FILE} 2>&1


elif [ "$1"x == "val"x ]; then
  # ${PYTHON} -u main.py --configs ${CONFIGS_TEST_DEPTH} --drop_last y \
  #                      --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
  #                      --phase test --gpu 3 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
  #                      --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
  #                      --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms_depth_v3_p8

  cd lib/metrics
  ${PYTHON} -u cityscapes_evaluator.py --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms_depth_v3_p8/label  \
                                       --gt_dir ${DATA_DIR}/val/label


elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/test --log_to_file n --out_dir test >> ${LOG_FILE} 2>&1

else
  echo "$1"x" is invalid..."
fi
