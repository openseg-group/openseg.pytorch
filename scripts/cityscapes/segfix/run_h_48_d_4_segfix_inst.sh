#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
nvidia-smi
export PYTHONPATH="$PWD":$PYTHONPATH
${PYTHON} -m pip install yacs
${PYTHON} -m pip install torchcontrib
${PYTHON} -m pip install pydensecrf
DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"

export sscrop=1

if [ -z $dt_max_distance ]; then
  export dt_max_distance=5
fi

echo dt_max_distance: $dt_max_distance

export dt_num_classes=8

echo dt_num_classes: $dt_num_classes

if [ -z $offset_dir ]; then
  offset_dir="offset_gt/dt_offset_inst_w_stuff"
fi

export offset_dir=$offset_dir
echo offset_dir: $offset_dir

DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"

BACKBONE="hrnet48"
CONFIGS="configs/cityscapes/H_SEGFIX.json"

MODEL_NAME="segfix_hrnet"
LOSS_TYPE="segfix_loss"
MAX_ITERS=20000

CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_${LOSS_TYPE}_inst_"$2
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="./pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --test_interval 2000 \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} \
                       --base_lr 0.04 \
                       2>&1 | tee ${LOG_FILE} 

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y --test_interval 1000 \
                       --phase train --gathered n --loss_balance y --log_to_file n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --gpu 0 1 2 3 \
                       --resume_continue y --resume ./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} \
                       2>&1 | tee -a ${LOG_FILE} 

elif [ "$1"x == "segfix_pred_val"x ]; then
  OUT_DIR=$PWD/segfix_pred/cityscapes/instance/offset_${BACKBONE}/
  mkdir -p ${OUT_DIR}

  if [ -z "$3" ]; then
    CKPT=./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth
  else
    CKPT=$3
  fi

  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test_offset --gpu 0 1 2 3 --resume ${CKPT} \
                       --log_to_file n --loss_type $LOSS_TYPE \
                       --out_dir ${OUT_DIR} \
                       test.eval_set val \
                       test.sscrop True

elif [ "$1"x == "segfix_pred_test"x ]; then
  OUT_DIR=$PWD/segfix_pred/cityscapes/test/instance/offset_${BACKBONE}/
  mkdir -p ${OUT_DIR}

  if [ -z "$3" ]; then
    CKPT=./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth
  else
    CKPT=$3
  fi

  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test_offset --gpu 0 1 2 3 --resume ${CKPT} \
                       --log_to_file n --loss_type $LOSS_TYPE \
                       --out_dir ${OUT_DIR} \
                       test.eval_set test \
                       test.sscrop True

elif [ "$1"x == "segfix_simple_pred_val"x ]; then
  export batch_size=4
  OUT_DIR=$PWD/segfix_pred/cityscapes/instance/offset_${BACKBONE}/
  mkdir -p ${OUT_DIR}

  if [ -z "$3" ]; then
    CKPT=./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth
  else
    CKPT=$3
  fi

  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test_offset --gpu 0 1 2 3 --resume ${CKPT} \
                       --log_to_file n --loss_type $LOSS_TYPE \
                       --out_dir ${OUT_DIR} \
                       test.eval_set val \
                       test.sscrop False

elif [ "$1"x == "segfix_simple_pred_test"x ]; then
  export batch_size=4
  OUT_DIR=$PWD/segfix_pred/cityscapes/test/instance/offset_${BACKBONE}/
  mkdir -p ${OUT_DIR}

  if [ -z "$3" ]; then
    CKPT=./checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth
  else
    CKPT=$3
  fi

  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test_offset --gpu 0 1 2 3 --resume ${CKPT} \
                       --log_to_file n --loss_type $LOSS_TYPE \
                       --out_dir ${OUT_DIR} \
                       test.eval_set test \
                       test.sscrop False

else
  echo "$1"x" is invalid..."
fi