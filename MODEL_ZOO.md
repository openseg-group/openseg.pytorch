# OCNet series

The following tables listed segmentation results on various datasets. To perform the validation, simply download and put checkpoints to corresponding directories, and run the script. For example, to evaluate `HRNet-W48 + OCR` on Cityscapes, you should download `ocr/Cityscapes/hrnet_w48_ocr_1_latest.pth` and put it under `~/checkpoints/cityscapes`, then run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to start validation.

## Cityscapes

Performance on the Cityscapes dataset. The models are trained and tested with input size of 512x1024 and 1024x2048 respectively. The performance of HRNet baseline is around 80.6% based on our training settings, where we train the models with smaller batch size and less iterations compared with the original setting.

Checkpoints should be put under `~/checkpoints/cityscapes`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | mIoU w/ SegFix | Link | Script |
| :---- | :------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
Base-OC | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.49 | 80.55 | [Log](https://drive.google.com/open?id=1bdO_yyuUH63fBP8AE0DO_9OJmJiuvPvw) / [Model](https://drive.google.com/open?id=1AyfnfIt_Aci3CoKup0uVY0UczJS3BiS7) | scripts/cityscapes/ocnet/run_r_101_d_8_baseoc_train.sh |
ISA | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.55 | 80.62 | [Log](https://drive.google.com/open?id=1gkWYJYSodnRcGrBQPYeDg47lsV9fiAhQ) / [Model](https://drive.google.com/open?id=1Sf9YFjo9dpirojzLev8CfHAc99U6cBwH) | scripts/cityscapes/isa/run_r_101_d_8_isa_train.sh |
OCR | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.63 | 80.68 | [Log](https://drive.google.com/open?id=1mKUM15UQXj5QYwvW6gJ6wQ0KDhJkWbFA) / [Model](https://drive.google.com/open?id=1bUCC3PEvuTBgfUpJlswEjdSJ_iGvg-Px) | scripts/cityscapes/ocrnet/run_r_101_d_8_ocrnet_train.sh |
ASP-OCR | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.89 | 80.69 | [Log](https://drive.google.com/open?id=1pT2OaCU6uGhNKH3TOJWvgvELYWV-0Fyd) / [Model](https://drive.google.com/open?id=1PXg7RK0LOOmTUNhjFOQXRswx0RAwCw2a) | scripts/cityscapes/ocrnet/run_r_101_d_8_asp_ocrnet_train.sh |
OCR | HRNet-W48 | Train | Val | 80000 | 8 | No | No | No | 81.09 | 81.73 | [Log](https://drive.google.com/open?id=1rHzUdSmLjvKsVkG-XpRzEpNX2zzU0hZc) / [Model](https://drive.google.com/open?id=1SJAgAhFODCqm_6L8KRkFFFr7dB2l6aC_) | scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh |

#### SegFix

On Cityscapes, we can use SegFix scheme to further refine the boundary of segmentation results. To apply SegFix, you should first download [offset_semantic.zip](https://drive.google.com/open?id=1iDP2scYmy51XJww-888oouNpRBksmrkv) to `$DATA_ROOT/cityscapes`, then unzip the archive. Take HRNet-W48 based OCR as an example. To refine the results on Cityscapes val set, you should first run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to obtain the baseline results, then run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh segfix 1 val` to apply SegFix.

## PASCAL-Context

The models are trained with the input size of 520x520, and tested with original size.

Checkpoints should be put under `~/checkpoints/pascal_context`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :---- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 60000 | 16 | No | No | No | 55.11 | [Log](https://drive.google.com/open?id=1cJcI3hL0MA4bxWQOCViV0J2ispIgYteV) / [Model](https://drive.google.com/open?id=1hJhlOFh2Vltuy8ebNVy3IX037UAptvgE) | scripts/pascal_context/run_h_48_d_4_ocr_train.sh |

## LIP

The models are trained and tested with input size of 473x473.

Checkpoints should be put under `~/checkpoints/lip`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :---- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 100000 | 32 | No | No | Yes | 56.72 | [Log](https://drive.google.com/open?id=1o6hOZWBJNk2LHxVJCtT7bW3u8SdHZb4f) / [Model](https://drive.google.com/open?id=1jlcJ_FwsadgxR1QrDw5Cxy2_9me86hUh) | scripts/lip/run_h_48_d_4_ocr_train.sh |

## COCO-Stuff

The models are trained with input size of 520x520, and tested with original size.

Checkpoints should be put under `~/checkpoints/coco_stuff`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :---- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 60000 | 16 | Yes | No | No | 39.61 | [Log](https://drive.google.com/open?id=1cfxEJFPsg_QaFU3nm5E4La_fIlgmSjr3) / [Model](https://drive.google.com/open?id=13tXkK9maID7ajSOxSsOavdqKHhFJ4HvD) | scripts/coco_stuff/run_h_48_d_4_ocr_ohem/train.sh |
OCR | HRNet-W48 | Train | Val | 60000 | 16 | Yes | Yes | Yes | 40.20 | same as above | scripts/coco_stuff/run_h_48_d_4_ocr_ohem_train.sh |

## ADE20K

The models are trained with input size of 520x520, and tested with original size.

Checkpoints should be put under `~/checkpoints/ade20k`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :---- | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 150000 | 16 | Yes | No | No | 44.62 | [Log](https://drive.google.com/open?id=1dfn9t5Pb1-IslO-_E8BHqm2qDTLfGMhy) / [Model](https://drive.google.com/open?id=1CAZBzNFh5DiUTT8KwlGV5YpQyFZn_C91) | scripts/ade20k/hrnet/run_h_48_d_4_ocr_ohem.sh |
OCR | HRNet-W48 | Train | Val | 150000 | 16 | Yes | Yes | Yes | 46.19 | same as above | scripts/ade20k/hrnet/run_h_48_d_4_ocr_ohem.sh |

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

Before starting training, you should download the corresponding ImageNet pretrained models to `pretrained_model/`. By default, we use HRNet-W48 (`hrnet48`) or Higher-HRNet (`hrnext20`) as backbone, but you can choose lighter ones by modifying `BACKBONE` and `PRETRAINED_MODEL` in the script.

| Backbone | Pretrained Model | 
| :---- | :----: |
| hrnet18 | [GoogleDrive](https://drive.google.com/file/d/1sLqUR30qG91km0vDmPGJLt8efP-vfiXX/view?usp=sharing) |
| hrnet32 | [GoogleDrive](https://drive.google.com/file/d/1Rjo3O0AAzL0LXBeGoR2qi3Tu9hT9KV9z/view?usp=sharing) |
| hrnet48 | [GoogleDrive](https://drive.google.com/file/d/1XyQMb2ZjAibqzumXCI30Na5Bl3Zt0bfn/view?usp=sharing) |
| hrnext20 | [GoogleDrive](https://drive.google.com/file/d/1vzUJHHZyH3GdtCnXVs4uK_9cDPxZdGLY/view?usp=sharing) |

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

You can download [offset_semantic.zip](https://drive.google.com/open?id=1iDP2scYmy51XJww-888oouNpRBksmrkv) or [offset_instance.zip](https://drive.google.com/open?id=1UXj6-XCXrPGAzDq3F1GGRpaF32nNTF4m) to `${DATA_ROOT}/cityscapes` and extract the archive.

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
| SegFix | HRNet-W48 | train | `scripts/cityscapes/segfix/run_h_48_d_4_segfix.sh`  | [GoogleDrive](https://drive.google.com/file/d/1PSTA5LetgqBUFFDPkvyuZk3ImjvvV1mL/view?usp=sharing) |
| SegFix | HRNet-W48 | train + val | `scripts/cityscapes/segfix/run_h_48_d_4_segfix_trainval.sh`  | - |
| SegFix | Higher-HRNet | train | `scripts/cityscapes/segfix/run_hx_20_d_2_segfix.sh`  | [GoogleDrive](https://drive.google.com/file/d/1CtNXrdu1PWesd9ZOMVk-svKaQNfh79mp/view?usp=sharing) |
| SegFix | Higher-HRNet | train + val | `scripts/cityscapes/segfix/run_hx_20_d_2_segfix_trainval.sh`  | [GoogleDrive](https://drive.google.com/file/d/1fJsIHmOFsRQLoP-TXUg2FjYQxittNU6u/view?usp=sharing) |


### Released prediction files

We have released the prediction of some state-of-the-arts approaches and their SegFixed results. The files can be found [here](https://drive.google.com/open?id=1ZTpzyGcjme7Cgz-PC6Urn27Qw5n29d9U). The performances are listed in the table below:

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
| SegFix | Higher-W48 | train | `scripts/cityscapes/segfix/run_h_48_d_4_segfix_inst.sh`  | - |
| SegFix | Higher-HRNet | train | `scripts/cityscapes/segfix/run_hx_20_d_2_segfix_inst.sh`  | [GoogleDrive](https://drive.google.com/file/d/1ehCJzzYNs-9mJs8lPr82Gddh59o6O8-E/view?usp=sharing) |

### Released prediction files

We have released the prediction of some state-of-the-arts approaches and their SegFixed results. The files can be found [here](https://drive.google.com/open?id=184RXq8-RT8cdt5ojQGa1ziGN5iOxU1Xh). The performances are listed in the table below:

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