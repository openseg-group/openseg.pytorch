#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../
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
  offset_dir="offset_gt/dt_offset"
fi

export offset_dir=$offset_dir
echo offset_dir: $offset_dir

BACKBONE="hrnet2x20"
CONFIGS="configs/segfix/H_SEGFIX.json"

MODEL_NAME="segfix_hrnet"
LOSS_TYPE="segfix_loss"
MAX_ITERS=100000
LR=0.04
BATCH_SIZE=16

CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_${LOSS_TYPE}_"$2
LOG_FILE="./log/segfix/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="./pretrained_model/hr_rnet_bt_w20_imagenet_pretrained.pth"

DATA_DIR="${DATA_ROOT}/cityscapes ${DATA_ROOT}/ade20k"
CHILD_CONFIGS="['configs/cityscapes/H_SEGFIX.json', 'configs/ade20k/H_SEGFIX.json']"

if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --base_lr $LR \
                       --train_batch_size $BATCH_SIZE \
                       --val_batch_size $BATCH_SIZE \
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
                       --test_interval 1000 \
                       \
                       child_config_files "${CHILD_CONFIGS}" \
                       use_adaptive_transform True \
                       2>&1 | tee ${LOG_FILE} 
                       

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --train_batch_size $BATCH_SIZE \
                       --val_batch_size $BATCH_SIZE \
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
                       --resume ./checkpoints/segfix/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --test_interval 1000 \
                       \
                       child_config_files "${CHILD_CONFIGS}" \
                       use_adaptive_transform True \
                       2>&1 | tee -a ${LOG_FILE} 

elif [ "$1"x == "test_offset"x ]; then
  if [ -z "$3" ]; then
    CKPT=./checkpoints/segfix/${CHECKPOINTS_NAME}_latest.pth
  else
    CKPT=$3
  fi

  OUT_DIR=$PWD/segfix_pred/cityscapes/semantic/offset_${BACKBONE}_joint/
  mkdir -p ${OUT_DIR}
  DATA_DIR="/msravcshare/dataset/cityscapes"
  CONFIGS="configs/cityscapes/H_48_D_4_DT_OFFSET.json"


  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test_offset --gpu 0 1 2 3 --resume .${CKPT} \
                      --log_to_file n --out_dir ${OUT_DIR} \
                       --loss_type $LOSS_TYPE --data_dir ${DATA_DIR} \
                       test.eval_set val \
                       test.sscrop True

  ####################################################################

  OUT_DIR=$PWD/segfix_pred/ade20k/semantic/offset_${BACKBONE}_joint/
  mkdir -p ${OUT_DIR}
  DATA_DIR="/msravcshare/dataset/ade20k"
  CONFIGS="configs/ade20k/H_48_D_4_DT_OFFSET.json"

  ${PYTHON} -u main.py --configs ${CONFIGS} --drop_last y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test_offset --gpu 0 1 2 3 --resume ${CKPT} \
                       --log_to_file n --out_dir ${OUT_DIR} \
                       --data_dir ${DATA_DIR} \
                       --loss_type $LOSS_TYPE \
                       val.data_transformer.size_mode diverse_size \
                       test.eval_set val \
                       test.sscrop False

else
  echo "$1"x" is invalid..."
fi
