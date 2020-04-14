#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile

# check the enviroment info
nvidia-smi
${PYTHON} -m pip install yacs
${PYTHON} -m pip install torchcontrib
${PYTHON} -m pip install pydensecrf

export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="${DATA_ROOT}/ade20k"
SAVE_DIR="${DATA_ROOT}/seg_result/ade20k/"
BACKBONE="hrnet48"
CONFIGS="configs/ade20k/H_48_D_4.json"
CONFIGS_TEST="configs/ade20k/H_48_D_4_TEST.json"

MODEL_NAME="hrnet48_isa"
LOSS_TYPE="fs_auxohemce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_ohem_"$2
PRETRAINED_MODEL="./pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"
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
                       --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       > ${LOG_FILE} 2>&1
                       

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
                        >> ${LOG_FILE} 2>&1


elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --phase debug --gpu 0 --log_to_file n > ${LOG_FILE} 2>&1

elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test \
                       --gpu 0 1 2 3 \
                       --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image \
                       --log_to_file n \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ss

  cd lib/metrics
  ${PYTHON} -u ade20k_evaluator.py --configs ../../${CONFIGS} \
                                   --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ss/label \
                                   --gt_dir ${DATA_DIR}/val/label 

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/test --log_to_file n --out_dir test >> ${LOG_FILE} 2>&1

else
  echo "$1"x" is invalid..."
fi