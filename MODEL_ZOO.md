# OCNet series

The following tables listed segmentation results on various datasets. To perform the validation, simply download and put checkpoints to corresponding directories, and run the script. For example, to evaluate `HRNet-W48 + OCR` on Cityscapes, you should download `ocr/Cityscapes/hrnet_w48_ocr_1_latest.pth` and put it under `~/checkpoints/cityscapes`, then run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to start validation.

HRNet-W48 (Paddle) means using the ImageNet pretrained weights converted from [PaddleClas](https://github.com/PaddlePaddle/PaddleClas). OCR+RMI means using [RMI](https://github.com/ZJULearning/RMI) loss.

The current released checkpoints are previously trained with Pytorch-0.4.1 and we will release the checkpoints trained with Pytorch-1.7 soon.

## Cityscapes

Performance on the Cityscapes dataset. The models are trained and tested with input size of 512x1024 and 1024x2048 respectively. The performance of HRNet baseline is around 80.6% based on our training settings, where we train the models with smaller batch size and less iterations compared with the original setting.

Checkpoints should be put under `~/checkpoints/cityscapes`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | mIoU w/ SegFix | Link | Script |
| :---- | :------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
Base-OC | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.49 | 80.55 | [Log](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/base_ocnet_deepbase_resnet101_dilated8_1.log) / [Model](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/base_ocnet_deepbase_resnet101_dilated8_1_latest.pth) | scripts/cityscapes/ocnet/run_r_101_d_8_baseoc_train.sh |
ISA | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.55 | 80.62 | [Log](https://drive.google.com/open?id=1gkWYJYSodnRcGrBQPYeDg47lsV9fiAhQ) / [Model](https://drive.google.com/open?id=1Sf9YFjo9dpirojzLev8CfHAc99U6cBwH) | scripts/cityscapes/isa/run_r_101_d_8_isa_train.sh |
OCR | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.63 | 80.68 | [Log](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/spatial_ocrnet_deepbase_resnet101_dilated8_1.log) / [Model](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/spatial_ocrnet_deepbase_resnet101_dilated8_1_latest.pth) | scripts/cityscapes/ocrnet/run_r_101_d_8_ocrnet_train.sh |
ASP-OCR | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.89 | 80.69 | [Log](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/spatial_asp_ocrnet_deepbase_resnet101_dilated8_1.log) / [Model](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/spatial_asp_ocrnet_deepbase_resnet101_dilated8_1_latest.pth) | scripts/cityscapes/ocrnet/run_r_101_d_8_asp_ocrnet_train.sh |
OCR | HRNet-W48 | Train | Val | 80000 | 8 | No | No | No | 81.09 | 81.73 | [Log](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/hrnet_w48_ocr_1.log) / [Model](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/hrnet_w48_ocr_1_latest.pth) | scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh |
OCR | HRNet-W48 (Paddle) | Train | Val | 40000 | 16 | No | No | No | 81.53 | 82.78 | [Log](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/cityscapes_hrnet_w48_ocr_rmi_paddle_lr2x_run1.log) / [Model](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/cityscapes_hrnet_w48_ocr_rmi_paddle_lr2x_run1_latest.pth) | scripts/cityscapes/hrnet/run_h_48_d_4_ocr_rmi_paddle.sh |
OCR+RMI | HRNet-W48 (Paddle) | Train | Val | 40000 | 16 | No | No | No | 82.57,82.64 | 83.20,83.22 | [Log](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/cityscapes_hrnet_w48_ocr_rmi_paddle_lr2x_run2.log) / [Model](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/cityscapes_hrnet_w48_ocr_rmi_paddle_lr2x_run2_latest.pth) | scripts/cityscapes/hrnet/run_h_48_d_4_ocr_rmi_paddle.sh |


### How to reproduce the HRNet + OCR with Mapillary pretraining
To help you to reproduce our best results on the Cityscapes leaderboard, we explain the details of the training pipeline as following:
* (1) We use the model `HRNet_W48_OCR_B` as the main architecture, which decreases the intput feature map channels from `720` to `256` (instead of `512`) w/o almost no performance drop.
* (2) We train the `HRNet_W48_OCR_B` on the original Mapillary training set with `batch size=16`, `crop size=1024x1024`, `base lr=0.01`, and `max iterations=500,000`
and achieve `50.8` on the Mapillary validation set. We have released the pretrained checkpoint [hrnet_w48_ocr_b_mapillary_bs16_500000_1024x1024_lr0.01_1_latest.pth](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/hrnet_w48_ocr_b_mapillary_bs16_500000_1024x1024_lr0.01_1_latest.pth).
* (3) We fine-tune the above Mapillary pretrained models on the Cityscapes `train + val` set with script `run_h_48_d_4_ocr_b_mapillary_trainval_ohem.sh`. Here we use smaller base learning rate `0.001`.
* (4) We fine-tune the models after (3) on the Cityscapes `coarse` set with script `run_h_48_d_4_ocr_b_mapillary_trainval_coarse_ohem.sh`. Here we also empirically find that freezing the BN statistics achieves slightly better results (+0.1%).
* (5) Last, we fine-tune the models on the Cityscapes `train + val` set with script `run_h_48_d_4_ocr_b_mapillary_trainval_coarse_trainval_ohem.sh`.
Finally, you could achieve the performance around `84.2%` on the Cityscapes leaderboard.
We have released the pretrained checkpoint [hrnet_w48_ocr_b_hrnet48_8_20000_trainval_coarse_trainval_mapillary_pretrain_freeze_bn_1_latest.pth](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.cityscapes/hrnet_w48_ocr_b_hrnet48_8_20000_trainval_coarse_trainval_mapillary_pretrain_freeze_bn_1_latest.pth).


### SegFix

On Cityscapes, we can use SegFix scheme to further refine the boundary of segmentation results. To apply SegFix, you should first download [offset_semantic.zip](https://drive.google.com/open?id=1iDP2scYmy51XJww-888oouNpRBksmrkv) to `$DATA_ROOT/cityscapes`, then unzip the archive. Take HRNet-W48 based OCR as an example. To refine the results on Cityscapes val set, you should first run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to obtain the baseline results, then run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh segfix 1 val` to apply SegFix.

## PASCAL-Context

The models are trained with the input size of 520x520, and tested with original size.

Checkpoints should be put under `~/checkpoints/pascal_context`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :---- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 60000 | 16 | No | No | No | 55.11 | [Log](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.pascal_context/hrnet_w48_ocr_hrnet48_2.log) / [Model](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.pascal_context/hrnet_w48_ocr_hrnet48_2_latest.pth) | scripts/pascal_context/run_h_48_d_4_ocr_train.sh |
OCR | HRNet-W48 (Paddle)  | Train | Val | 60000 | 16 | No | No | No | 57.82 | [Log]() / [Model]() | scripts/pascal_context/run_h_48_d_4_ocr_train_paddle.sh |
OCR | HRNet-W48 (Paddle)  | Train | Val | 60000 | 16 | No | Yes | Yes | 59.13 | [Log]() / [Model]() | scripts/pascal_context/run_h_48_d_4_ocr_train_paddle.sh |
OCR+RMI | HRNet-W48 (Paddle)  | Train | Val | 60000 | 16 | No | No | No | 58.53,58.72 | [Log](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/pascal_context_hrnet_w48_ocr_rmi_paddle_run1.log) / [Model](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/pascal_context_hrnet_w48_ocr_rmi_paddle_run1_latest.pth) | scripts/pascal_context/run_h_48_d_4_ocr_train_rmi_paddle.sh |
OCR+RMI | HRNet-W48 (Paddle)  | Train | Val | 60000 | 16 | No | Yes | Yes | 59.62,59.86 | [Log](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/pascal_context_hrnet_w48_ocr_rmi_paddle_run2.log) / [Model](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/pascal_context_hrnet_w48_ocr_rmi_paddle_run2_latest.pth) | scripts/pascal_context/run_h_48_d_4_ocr_train_rmi_paddle.sh |

## LIP

The models are trained and tested with input size of 473x473.

Checkpoints should be put under `~/checkpoints/lip`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :---- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 100000 | 32 | No | No | Yes | 56.72 | [Log](https://github.com/hsfzxjy/models.storage/releases/download/opeseg.pytorch.lip/hrnet_w48_ocr_1.log) / [Model](https://github.com/hsfzxjy/models.storage/releases/download/opeseg.pytorch.lip/hrnet_w48_ocr_1_latest.pth) | scripts/lip/run_h_48_d_4_ocr_train.sh |
OCR | HRNet-W48 | Train | Val | 100000 | 32 | No | No | Yes | 57.87 | [Log]() / [Model]() | scripts/lip/run_h_48_d_4_ocr_train_paddle.sh |
OCR+RMI | HRNet-W48 | Train | Val | 100000 | 32 | No | No | Yes | 58.21 | [Log](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/lip_hrnet_w48_ocr_rmi_paddle.log) / [Model](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/lip_hrnet_w48_ocr_rmi_paddle_latest.pth) | scripts/lip/run_h_48_d_4_ocr_train_rmi_paddle.sh |


## COCO-Stuff

The models are trained with input size of 520x520, and tested with original size.

Checkpoints should be put under `~/checkpoints/coco_stuff`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :---- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 60000 | 16 | Yes | No | No | 39.61 | [Log](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.coco_stuff/hrnet_w48_ocr_hrnet48_ohem_2.log) / [Model](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.coco_stuff/hrnet_w48_ocr_hrnet48_ohem_2_latest.pth) | scripts/coco_stuff/run_h_48_d_4_ocr_ohem/train.sh |
OCR | HRNet-W48 | Train | Val | 60000 | 16 | Yes | Yes | Yes | 40.20 | same as above | scripts/coco_stuff/run_h_48_d_4_ocr_ohem_train.sh |
OCR | HRNet-W48 (Paddle) | Train | Val | 60000 | 16 | Yes | No | No | 42.50 | [Log]() / [Model]() | scripts/coco_stuff/run_h_48_d_4_ocr_ohem_train_paddle.sh |
OCR | HRNet-W48 (Paddle) | Train | Val | 60000 | 16 | Yes | Yes | Yes | 43.26 | same as above | scripts/coco_stuff/run_h_48_d_4_ocr_ohem_train_paddle.sh |
OCR+RMI | HRNet-W48 (Paddle) | Train | Val | 60000 | 16 | Yes | No | No | 43.95 | [Log]() / [Model]() | scripts/coco_stuff/run_h_48_d_4_ocr_ohem_train_rmi_paddle.sh |
OCR+RMI | HRNet-W48 (Paddle) | Train | Val | 60000 | 16 | Yes | Yes | Yes | 45.20 | same as above | scripts/coco_stuff/run_h_48_d_4_ocr_ohem_train_rmi_paddle.sh |

## ADE20K

The models are trained with input size of 520x520, and tested with original size.

Checkpoints should be put under `~/checkpoints/ade20k`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :---- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 150000 | 16 | Yes | No | No | 44.62 | [Log](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.ade20k/hrnet_w48_ocr_hrnet48_ohem_1.log) / [Model](https://github.com/hsfzxjy/models.storage/releases/download/openseg.pytorch.ade20k/hrnet_w48_ocr_hrnet48_ohem_1_latest.pth) | scripts/ade20k/hrnet/run_h_48_d_4_ocr_ohem.sh |
OCR | HRNet-W48 | Train | Val | 150000 | 16 | Yes | Yes | Yes | 46.19 | same as above | scripts/ade20k/hrnet/run_h_48_d_4_ocr_ohem.sh |
OCR | HRNet-W48 (Paddle) | Train | Val | 150000 | 16 | Yes | No | No | --- | [Log]() / [Model]() | scripts/ade20k/hrnet/run_h_48_d_4_ocr_ohem_paddle.sh |
OCR | HRNet-W48 (Paddle) | Train | Val | 150000 | 16 | Yes | Yes | Yes | --- | same as above | scripts/ade20k/hrnet/run_h_48_d_4_ocr_ohem_paddle.sh |
OCR+RMI | HRNet-W48 (Paddle) | Train | Val | 150000 | 16 | Yes | No | No | 46.59 | [Log](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/ade20k_hrnet_w48_ocr_hrnet48_rmi_paddle.log) / [Model](https://github.com/openseg-group/openseg.pytorch/releases/download/v1.0/ade20k_hrnet_w48_ocr_hrnet48_rmi_paddle_latest.pth) | scripts/ade20k/hrnet/run_h_48_d_4_ocr_rmi_paddle.sh |
OCR+RMI | HRNet-W48 (Paddle) | Train | Val | 150000 | 16 | Yes | Yes | Yes | 47.98 | same as above | scripts/ade20k/hrnet/run_h_48_d_4_ocr_rmi_paddle.sh |


# SegFix

**We strongly recommend you to use our SegFix to improve your segmentation results as it is super easy & fast to use.**

SegFix is a general effective (model-agnostic) post-processing scheme (kinds of like DenseCRF). The key idea of the SegFix is to replace the labels of the boundary pixels with the label of the interior pixels. SegFix can be used to improve the semantic/instance segmentation results of any existing approaches, e.g., HRNet, DeepLabv3, OCR, PointRend, MaskRCNN, without any re-training or fine-tuning. We have made the inference code and the offset files of our SegFix method. Please try our SegFix in your Cityscapes submission and you can achieve much better performance. As illustrated in the followed examples, our SegFix is complementary with various very recent methods, such as the PointRend by FAIR.

## SegFix Pipelines

Currently openseg allows users to use SegFix in the following ways.

For whom want to try SegFix by training a new SegFix model, you should:

 1. Generate ground truth offsets.
 2. Download ImageNet pretrained model to `pretrained_model/`.
 3. Run the training script.
 4. Run the prediction script to predict offsets for Cityscapes val / test set.
 5. Run the refinement script to refine any labels with the offsets.

For whom want to try SegFix by using a pretrained SegFix model, you should:

 1. Download corresponding checkpoints to `checkpoints/cityscapes/`.
 2. Run the prediction script to predict offsets for Cityscapes val / test set.
 3. Run the refinement script to refine any labels with the offsets.

For whom want to try SegFix by using offline-generated offsets, you should:

 1. Download corresponding offsets files to `${DATA_ROOT}/cityscapes` and extract them.
 2. Run the refinement script to refine any labels with the offsets.

More details are introduced in the following sections.

### Training

```bash
# Training
bash scripts/cityscapes/segfix/<script>.sh train 1
```

Before starting training, you should download the corresponding ImageNet pretrained models to `pretrained_model/`. By default, we use HRNet-W48 (`hrnet48`) or HRNet2x-W20 (`hrnet2x20`) as backbone, but you can choose lighter ones by modifying `BACKBONE` and `PRETRAINED_MODEL` in the script.

| Backbone | Pretrained Model | 
| :---- | :----: |
| hrnet18 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/openseg-pytorch-pretrained/hrnetv2_w18_imagenet_pretrained.pth) |
| hrnet32 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/openseg-pytorch-pretrained/hrnetv2_w32_imagenet_pretrained.pth) |
| hrnet48 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/openseg-pytorch-pretrained/hrnetv2_w48_imagenet_pretrained.pth) |
| hrnet2x20 | [Github](https://github.com/hsfzxjy/models.storage/releases/download/openseg-pytorch-pretrained/hr_rnet_bt_w20_imagenet_pretrained.pth) |

### Prediction

SegFix generates a kind of intermediate files called **offsets**, which can be used to refine segmentation results from any models.

```bash
# Predict offsets for val set and save to path
# `segfix_pred/val/[semantic | instance]/cityscapes/
bash scripts/cityscapes/segfix/<script>.sh segfix_pred_val 1
# Predict offsets for test set and save to path
# `segfix_pred/test/[semantic | instance]/cityscapes/
bash scripts/cityscapes/segfix/<script>.sh segfix_pred_test 1
```

For example, running 
```bash
bash scripts/cityscapes/segfix/run_h_48_d_4_segfix.sh segfix_pred_val 1
```
will store offsets to `offset_pred/val/semantic/cityscapes/offset_hrnet48/`. Then you can run
```bash
python scripts/cityscapes/segfix.py \
  --offset offset_pred/val/semantic/cityscapes/offset_hrnet48/ \
  --input <your labels>
```
to refine your own labels.

### Use offline-generated offsets

You can download [offset_semantic.zip](https://github.com/hsfzxjy/models.storage/releases/download/segfix.offsets/offset_semantic.zip) or [offset_instance.zip](https://github.com/hsfzxjy/models.storage/releases/download/segfix.offsets/offset_instance.zip) to `${DATA_ROOT}/cityscapes` and extract the archive.

### Refinement

You can use `scripts/cityscapes/segfix.py` (semantic) or `scripts/cityscapes/segfix_instance.py` (instance) to apply SegFix on your own label files. Usage:
```bash
python scripts/cityscapes/segfix[_instance].py \
  --input <path/to/your/label/dir> \
  --split <SPLIT> \
  [ --offset <OFFSET_DIR>] \
  [ --out <OUT_DIR>]
```
where 
  + `<SPLIT>` is `test` or `val`.
  + `<OFFSET_DIR>` is the location of SegFix offsets, default to `$DATA_ROOT/cityscapes/val/offset_pred/[semantic | instance]/offset_hrnext/` or `$DATA_ROOT/cityscapes/test_offset/[semantic | instance]/offset_hrnext/`.
  + `<OUT_DIR>` is an optional output directory.


## Cityscapes Semantic Segmentation

### Generating ground truth for SegFix

Simply run

```bash
python lib/datasets/preprocess/cityscapes/dt_offset_generator.py
```

### Scripts and checkpoints for SegFix models

| Method | Backbone | Train Set | Script | Checkpoint | 
| :----: | :----: | :--: | :--: | :--: |
| SegFix | HRNet-W48 | train | `scripts/cityscapes/segfix/run_h_48_d_4_segfix.sh`  | [Github](https://github.com/hsfzxjy/models.storage/releases/download/segfix.pretrained/segfix_hrnet_hrnet48_segfix_loss_iter80000_1_latest.pth) |
| SegFix | HRNet-W48 | train + val | `scripts/cityscapes/segfix/run_h_48_d_4_segfix_trainval.sh`  | - |
| SegFix | HRNet2x-W20 | train | `scripts/cityscapes/segfix/run_hx_20_d_2_segfix.sh`  | [Github](https://github.com/hsfzxjy/models.storage/releases/download/segfix.pretrained/segfix_hrnet_hrnext20_segfix_loss_1_latest.pth) |
| SegFix | HRNet2x-W20 | train + val | `scripts/cityscapes/segfix/run_hx_20_d_2_segfix_trainval.sh`  | [Github](https://github.com/hsfzxjy/models.storage/releases/download/segfix.pretrained/segfix_hrnet_hrnext20_segfix_loss_trainval_1_latest.pth) |


### Released prediction files

We have released the prediction of some state-of-the-arts approaches and their SegFixed results. The files can be found [here](https://github.com/hsfzxjy/models.storage/releases/tag/segfix.prediction). The performances are listed in the table below:

| Method | Test Set | mIoU w/o SegFix | mIoU w/ SegFix |
| :---- | :----: | :--: | :--: |
| HRNet-W48 | val | 81.1 | 81.6 |
| HRNet-W48 + OCR | test | 84.2 | 84.5 |

### Refinement with SegFix

Except for the refinement scripts mentioned above, several scripts in openseg have built-in support for SegFix post-processing. For example, you can first run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to get the baseline prediction of HRNet-W48 + OCR model, then run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh segfix 1 val` to further apply SegFix on the prediction labels.

## Cityscapes Instance Segmentation

### Generating ground truth for SegFix

Simply run

```bash
python lib/datasets/preprocess/cityscapes/instance_dt_offset_generator.py
```

### Scripts and checkpoints for SegFix models

| Method | Backbone | Train Set | Script | Checkpoint | 
| :----: | :----: | :--: | :--: | :--: |
| SegFix | HRNet-W48 | train | `scripts/cityscapes/segfix/run_h_48_d_4_segfix_inst.sh`  | - |
| SegFix | HRNet2x-W20 | train | `scripts/cityscapes/segfix/run_hx_20_d_2_segfix_inst.sh`  | [Github](https://github.com/hsfzxjy/models.storage/releases/download/segfix.pretrained/segfix_hrnet_hrnext20_segfix_loss_inst_1_latest.pth) |

### Released prediction files

We have released the prediction of some state-of-the-arts approaches and their SegFixed results. The files can be found [here](https://github.com/hsfzxjy/models.storage/releases/tag/segfix.prediction). The performances are listed in the table below:

| Method | Test Set | AP w/o SegFix | AP w/ SegFix |
| :---- | :----: | :--: | :--: |
| MaskRCNN (w/ COCO, Detectron2) | val | 36.5 | 38.2 |
| PointRend (w/ COCO, Detectron2) | val | 37.9 | 39.5 |
| MaskRCNN (w/ COCO, Detectron2) | test | 32.0 | 33.3 |
| PointRend (w/ COCO, Detectron2) | test | 33.3 | 34.8 |
| PANet (w/ COCO) | test | 36.4 | 37.8 |
| PolyTransform (w/ COCO) | test | 40.1 | 41.2 |

### Refinement with SegFix

Simply follow the instruction in the **Refinement** section.
