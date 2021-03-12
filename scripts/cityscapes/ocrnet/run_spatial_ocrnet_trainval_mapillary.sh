#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
nvidia-smi
export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"
BACKBONE="deepbase_resnet101_dilated8"
CONFIGS="configs/cityscapes/${BACKBONE}_ohem.json"
CONFIGS_TEST="configs/cityscapes/${BACKBONE}_test.json"

MODEL_NAME="spatial_ocrnet"
LOSS_TYPE="fs_auxohemce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_trainval_mapillary_"$2
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`


MAX_ITERS=50000
PRETRAINED_MODEL="./checkpoints/cityscapes/spatial_ocrnet_deepbase_resnet101_dilated8_trainval_1_latest.pth"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --only_mapillary y \
                       --base_lr 0.0001 \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --resume ${PRETRAINED_MODEL} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       2>&1 | tee ${LOG_FILE}


elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --only_mapillary y --base_lr 0.0001 \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --gpu 0 1 2 3 \
                       --resume_continue y --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                        2>&1 | tee -a ${LOG_FILE}



elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --phase debug --gpu 0 --log_to_file n  2>&1 | tee ${LOG_FILE}

elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --log_to_file n --out_dir val 2>&1 | tee -a ${LOG_FILE}
  cd lib/metrics
  ${PYTHON} -u cityscapes_evaluator.py --pred_dir ../../results/cityscapes/test_dir/${CHECKPOINTS_NAME}/val/label \
                                       --gt_dir ${DATA_DIR}/val/label  >> "../../"${LOG_FILE} 2>&1


elif [ "$1"x == "test"x ]; then
  if [ "$3"x == "ss"x ]; then
    echo "[single scale] test"
    ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    ${PYTHON} -u main.py --configs ${CONFIGS_TEST} --drop_last y \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 1 2 3 --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi

else
  echo "$1"x" is invalid..."
fi
