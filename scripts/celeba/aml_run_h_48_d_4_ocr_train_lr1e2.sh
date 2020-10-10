#!/usr/bin/env bash

# $1 code path
# $2 dataset path
# $3 train or test
# $4 log_suffix

PYTHON="/opt/conda/bin/python"
${PYTHON} -c "import torch; print(torch.__version__)"

${PYTHON} -m pip install yacs

export PYTHONPATH=$1:$PYTHONPATH

DATA_DIR="$2/face_parse/CelebAMask-HQ"
SAVE_DIR="$2/seg_result/celeba/"
BACKBONE="hrnet48"

CONFIGS="configs/celeba/H_48_D_4.json"
CONFIGS_TEST="configs/celeba/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48_ocr"
LOSS_TYPE="fs_auxce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_"$4
LOG_FILE="./log/celeba/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
PRETRAINED_MODEL="./pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=150000

if [ "$3"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --base_lr 0.01 \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       2>&1 | tee ${LOG_FILE}


elif [ "$3"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --gpu 0 1 2 3 \
                       --resume_continue y \
                       --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                        2>&1 | tee -a ${LOG_FILE}


elif [ "$3"x == "val"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS_TEST} \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test \
                       --gpu 0 1 2 3 \
                       --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image \
                       --log_to_file n \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

  cd lib/metrics
  ${PYTHON} -u ade20k_evaluator.py --configs ../../${CONFIGS_TEST} \
                                   --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label \
                                   --gt_dir ${DATA_DIR}/val/label  


elif [ "$3"x == "test"x ]; then
  if [ "$5"x == "ss"x ]; then
    echo "[single scale] test"
    ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    ${PYTHON} -u main.py --configs ${CONFIGS_TEST} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi


else
  echo "$3"x" is invalid..."
fi
