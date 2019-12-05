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
/path/to/data/
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


# Segmentation Results

The following tables listed segmentation results on various datasets. To perform the validation, simply download and put checkpoints to corresponding directories, and run the script. For example, to evaluate `HRNet-W48 + OCR` on Cityscapes, you should download `ocnet/Cityscapes/hrnet_w48_ocr_1_latest.pth` and put it under `~/checkpoints/cityscapes`, then go to `scripts/cityscapes/hrnet/` and run `bash run_h_48_d_4_ocr.sh val 1` to start validation.

## Cityscapes

Performance on the Cityscapes dataset. The models are trained and tested with input size of 512x1024 and 1024x2048 respectively.

Checkpoints should be put under `~/checkpoints/cityscapes`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :----: | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
Base-OC | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.49 |  | scripts/cityscapes/ocnet/run_r_101_d_8_baseoc_train.sh |
OCR | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.63 |  | scripts/cityscapes/ocrnet/run_r_101_d_8_ocrnet_train.sh |
ASP-OCR | ResNet-101 | Train | Val | 40000 | 8 | No | No | No | 79.89 |  | scripts/cityscapes/ocrnet/run_r_101_d_8_asp_ocrnet_train.sh |
OCR | HRNet-W48 | Train | Val | 80000 | 8 | No | No | No | 81.09 |  | scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh |

## PASCAL-Context

The models are trained with the input size of 520x520, and tested with original size.

Checkpoints should be put under `~/checkpoints/pascal_context`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :----: | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 60000 | 16 | No | No | No | 55.11 |  | scripts/pascal_context/run_h_48_d_4_ocr_train.sh |

## LIP

The models are trained and tested with input size of 473x473.

Checkpoints should be put under `~/checkpoints/lip`.

Methods | Backbone | Train Set | Test Set | Iterations | Batch Size | OHEM | Multi-scale | Flip | mIoU | Link | Script |
| :----: | :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
OCR | HRNet-W48 | Train | Val | 100000 | 32 | No | No | Yes | 56.72 |  | scripts/lip/run_h_48_d_4_ocr_train.sh |