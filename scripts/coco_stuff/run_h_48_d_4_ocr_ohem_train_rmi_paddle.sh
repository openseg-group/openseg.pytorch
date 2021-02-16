#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../
. config.profile

# PYTHON="/data/anaconda/envs/pytorch1.7.1/bin/python"
# DATA_ROOT="/home/yuhui/teamdrive/dataset"
# DATA_ROOT=$3

${PYTHON} -m pip install yacs
${PYTHON} -m pip install torchcontrib
${PYTHON} -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="${DATA_ROOT}/coco_stuff_10k"
SAVE_DIR="${DATA_ROOT}/seg_result/coco_stuff/"
BACKBONE="hrnet48"

CONFIGS="configs/coco_stuff/H_48_D_4_RMI.json"
CONFIGS_TEST="configs/coco_stuff/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48_ocr"
LOSS_TYPE="fs_aux_rmi_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_ohem_rmi_paddle_"$2
LOG_FILE="./log/coco_stuff/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
PRETRAINED_MODEL="./pretrained_model/HRNet_W48_C_ssld_pretrained.pth"
MAX_ITERS=60000
BATCH_SIZE=16

if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --nbb_mult 10 \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 2 3 4 5 6 7 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       --train_batch_size ${BATCH_SIZE} \
                       --distributed \
                       --test_interval 10000 \
                       2>&1 | tee ${LOG_FILE}
                       

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --nbb_mult 10 \
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


elif [ "$1"x == "val"x ]; then
  if [ "$3"x == "ss"x ]; then
    ${PYTHON} -u main.py --configs ${CONFIGS} \
                          --data_dir ${DATA_DIR} \
                          --backbone ${BACKBONE} \
                          --model_name ${MODEL_NAME} \
                          --checkpoints_name ${CHECKPOINTS_NAME} \
                          --phase test \
                          --gpu 0 1 2 3 \
                          --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
                          --test_dir ${DATA_DIR}/val/image \
                          --log_to_file n \
                          --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ss

    cd lib/metrics
    ${PYTHON} -u cocostuff_evaluator.py --configs ../../${CONFIGS} \
                                      --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ss/label \
                                      --gt_dir ${DATA_DIR}/val/label
  else
    ${PYTHON} -u main.py --configs ${CONFIGS_TEST} \
                          --data_dir ${DATA_DIR} \
                          --backbone ${BACKBONE} \
                          --model_name ${MODEL_NAME} \
                          --checkpoints_name ${CHECKPOINTS_NAME} \
                          --phase test \
                          --gpu 0 1 2 3 4 5 6 7 \
                          --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
                          --test_dir ${DATA_DIR}/val/image \
                          --log_to_file n \
                          --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

    cd lib/metrics
    ${PYTHON} -u cocostuff_evaluator.py --configs ../../${CONFIGS_TEST} \
                                      --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label \
                                      --gt_dir ${DATA_DIR}/val/label 
  fi

elif [ "$1"x == "test"x ]; then
  if [ "$3"x == "ss"x ]; then
    echo "[single scale] test"
    ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/val/image --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    ${PYTHON} -u main.py --configs ${CONFIGS_TEST} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/val/image --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi

else
  echo "$1"x" is invalid..."
fi
