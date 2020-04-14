#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
nvidia-smi${PYTHON} -m pip install yacs

export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"
BACKBONE="hrnet48"
CONFIGS="configs/cityscapes/${BACKBONE}.json"

MAX_ITERS=80000
BATCH_SIZE=8

MODEL_NAME="hrnet48_spatial_ocr"
LOSS_TYPE="fs_auxohemce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_${BN_TYPE}_${BATCH_SIZE}_${MAX_ITERS}_OHEM09_"$2
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"

PRETRAINED_MODEL="./pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --train_batch_size ${BATCH_SIZE}\
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth --resume_val y \
                       # --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} \
                       # > ${LOG_FILE} 2>&1
                       

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --train_batch_size ${BATCH_SIZE} \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --gpu 0 1 2 3 \
                       --resume_continue y --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} \
                        >> ${LOG_FILE} 2>&1

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --phase debug --gpu 0 --log_to_file n  > ${LOG_FILE} 2>&1

elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --train_batch_size ${BATCH_SIZE} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --log_to_file n --out_dir val >> ${LOG_FILE} 2>&1
  cd lib/metrics
  ${PYTHON} -u cityscapes_evaluator.py --pred_dir ../../results/cityscapes/test_dir/${CHECKPOINTS_NAME}/val/label \
                                       --gt_dir ${DATA_DIR}/val/label  >> "../../"${LOG_FILE} 2>&1

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/test --log_to_file n --out_dir test >> ${LOG_FILE} 2>&1

else
  echo "$1"x" is invalid..."
fi
