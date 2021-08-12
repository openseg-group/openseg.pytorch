#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile

# DATA_ROOT=$3
# check the enviroment info
nvidia-smi
export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="${DATA_ROOT}/ade20k"
SAVE_DIR="${DATA_ROOT}/seg_result/ade20k/"
BACKBONE="hrnet48"
CONFIGS="configs/ade20k/H_48_D_4_RMI.json"
CONFIGS_TEST="configs/ade20k/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48_ocr"
LOSS_TYPE="fs_aux_rmi_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_rmi_paddle_$(date +%F_%H-%M-%S)"
PRETRAINED_MODEL="./pretrained_model/HRNet_W48_C_ssld_pretrained.pth"
MAX_ITERS=150000

LOG_FILE="./log/ade20k/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
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
                       --distributed \
                       --test_interval 10000 \
                       2>&1 | tee ${LOG_FILE}
                       

elif [ "$1"x == "resume"x ]; then
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
                       --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                        2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --phase debug --gpu 0 --log_to_file n 2>&1 | tee ${LOG_FILE}

elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS_TEST} \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test \
                       --gpu 0 1 2 3 4 5 6 7 \
                       --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image \
                       --log_to_file n \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

    # ${PYTHON} -u main.py --configs ${CONFIGS_TEST} \
    #                       --data_dir ${DATA_DIR} \
    #                       --backbone ${BACKBONE} \
    #                       --model_name ${MODEL_NAME} \
    #                       --checkpoints_name ${CHECKPOINTS_NAME} \
    #                       --phase test \
    #                       --gpu 0 1 2 3 4 5 6 7 \
    #                       --resume ./checkpoints/coco_stuff/${CHECKPOINTS_NAME}_latest.pth \
    #                       --test_dir ${DATA_DIR}/val/image \
    #                       --log_to_file n \
    #                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

  cd lib/metrics
  ${PYTHON} -u ade20k_evaluator.py --configs ../../${CONFIGS_TEST} \
                                   --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label \
                                   --gt_dir ${DATA_DIR}/val/label

else
  echo "$1"x" is invalid..."
fi
