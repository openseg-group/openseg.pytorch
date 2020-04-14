# Data Preparation

You need to download [Cityscapes](https://www.cityscapes-dataset.com/), [LIP](http://sysu-hcp.net/lip/) and [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/) datasets.

We arrange images and labels in another way. You could preprocess the files by running:

```bash
python lib/datasets/preprocess/cityscapes/cityscapes_generator.py --coarse True \
  --save_dir <path/to/preprocessed_cityscapes> --ori_root_dir <path/to/original_cityscapes>
python lib/datasets/preprocess/pascal_context/pascal_context_generator.py \
  --save_dir <path/to/preprocessed_context> --ori_root_dir <path/to/original_context>
# TODO: LIP Preprocess
```

and finally, the dataset directory should look like:

```
$DATA_ROOT
├── cityscapes
│   ├── coarse
│   │   ├── image
│   │   ├── instance
│   │   └── label
│   ├── train
│   │   ├── image
│   │   └── label
│   ├── val
│   │   ├── image
│   │   └── label
├── pascal_context
│   ├── train
│   │   ├── image
│   │   └── label
│   ├── val
│   │   ├── image
│   │   └── label
├── lip
│   ├── atr
│   │   ├── edge
│   │   ├── image
│   │   └── label
│   ├── cihp
│   │   ├── image
│   │   └── label
│   ├── train
│   │   ├── edge
│   │   ├── image
│   │   └── label
│   ├── val
│   │   ├── edge
│   │   ├── image
│   │   └── label
```

# Configuration

Before executing any scripts, your should first fill up the config file `config.profile` at project root directory. There are two items you should specify:

 + `PYTHON`, identifying your python executable.
 + `DATA_ROOT`, the root directory of your data. It should be the parent directory of `cityscapes`.

# Segmentation Results

The following tables listed segmentation results on various datasets. To perform the validation, simply download and put checkpoints to corresponding directories, and run the script. For example, to evaluate `HRNet-W48 + OCR` on Cityscapes, you should download `ocr/Cityscapes/hrnet_w48_ocr_1_latest.pth` and put it under `~/checkpoints/cityscapes`, then run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to start validation.

## Cityscapes

Performance on the Cityscapes dataset. The models are trained and tested with input size of 512x1024 and 1024x2048 respectively.

Checkpoints should be put under `~/checkpoints/cityscapes`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | mIoU w/ SegFix | Link | Script |
| :----: | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
Base-OC | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.49 | 80.55 | [Log](https://drive.google.com/open?id=1bdO_yyuUH63fBP8AE0DO_9OJmJiuvPvw) / [Model](https://drive.google.com/open?id=1AyfnfIt_Aci3CoKup0uVY0UczJS3BiS7) | scripts/cityscapes/ocnet/run_r_101_d_8_baseoc_train.sh |
OCR | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.63 | 80.68 | [Log](https://drive.google.com/open?id=1mKUM15UQXj5QYwvW6gJ6wQ0KDhJkWbFA) / [Model](https://drive.google.com/open?id=1bUCC3PEvuTBgfUpJlswEjdSJ_iGvg-Px) | scripts/cityscapes/ocrnet/run_r_101_d_8_ocrnet_train.sh |
ASP-OCR | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.89 | 80.69 | [Log](https://drive.google.com/open?id=1pT2OaCU6uGhNKH3TOJWvgvELYWV-0Fyd) / [Model](https://drive.google.com/open?id=1PXg7RK0LOOmTUNhjFOQXRswx0RAwCw2a) | scripts/cityscapes/ocrnet/run_r_101_d_8_asp_ocrnet_train.sh |
OCR | HRNet-W48 | Train | Val | 80000 | 8 | No | No | No | 81.09 | 81.73 | [Log](https://drive.google.com/open?id=1rHzUdSmLjvKsVkG-XpRzEpNX2zzU0hZc) / [Model](https://drive.google.com/open?id=1SJAgAhFODCqm_6L8KRkFFFr7dB2l6aC_) | scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh |

### SegFix

On Cityscapes, we can use SegFix scheme to further refine the boundary of segmentation results. To apply SegFix, you should first download [offset_semantic.zip](https://drive.google.com/open?id=1iDP2scYmy51XJww-888oouNpRBksmrkv) to `$DATA_ROOT/cityscapes`, then unzip the archive. Take HRNet-W48 based OCR as an example. To refine the results on Cityscapes val set, you should first run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to obtain the baseline results, then run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh segfix 1 val` to apply SegFix.

## PASCAL-Context

The models are trained with the input size of 520x520, and tested with original size.

Checkpoints should be put under `~/checkpoints/pascal_context`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :----: | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 60000 | 16 | No | No | No | 55.11 | [Log](https://drive.google.com/open?id=1cJcI3hL0MA4bxWQOCViV0J2ispIgYteV) / [Model](https://drive.google.com/open?id=1hJhlOFh2Vltuy8ebNVy3IX037UAptvgE) | scripts/pascal_context/run_h_48_d_4_ocr_train.sh |

## LIP

The models are trained and tested with input size of 473x473.

Checkpoints should be put under `~/checkpoints/lip`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :----: | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 100000 | 32 | No | No | Yes | 56.72 | [Log](https://drive.google.com/open?id=1o6hOZWBJNk2LHxVJCtT7bW3u8SdHZb4f) / [Model](https://drive.google.com/open?id=1jlcJ_FwsadgxR1QrDw5Cxy2_9me86hUh) | scripts/lip/run_h_48_d_4_ocr_train.sh |