#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
nvidia-smi
${PYTHON} -m pip install yacs
${PYTHON} -m pip install torchcontrib
${PYTHON} -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="${DATA_ROOT}/mapillary-vista-v1.1"
SAVE_DIR="${DATA_ROOT}/seg_result/mapillary/"
BACKBONE="hrnet48"

CONFIGS="configs/mapillary/H_48_D_4_1024x1024.json"

MODEL_NAME="hrnet_w48_ocr_b"
LOSS_TYPE="fs_auxce_loss"
LOG_FILE="./log/mapillary/${CHECKPOINTS_NAME}.log"
LOG_FILE="./log/mapillary/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="./pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=500000
BATCH_SIZE=16

if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
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
                       --train_batch_size ${BATCH_SIZE}
                       --base_lr 0.02 \
                       --test_interval 10000 \
                       2>&1 | tee ${LOG_FILE}


elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --include_val y  \
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
                       --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --train_batch_size ${BATCH_SIZE}
                        2>&1 | tee -a ${LOG_FILE}