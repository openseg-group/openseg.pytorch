# SegFix

SegFix is a novel yet simple model-agnostic post-processing scheme. Our model-agnostic post-processing scheme is a new work under progress, which can be applied to improve the results of any existing approaches without any re-training or fine-tuning.

SegFix can be applied on both Semantic Segmentation and Instance Segmentation tasks.

## SegFix for Semantic Segmentation

### Released prediction files

We have released the prediction of some state-of-the-arts approaches and their SegFixed results. The files can be found [here](https://drive.google.com/open?id=1ZTpzyGcjme7Cgz-PC6Urn27Qw5n29d9U). The performances are listed in the table below:

| Method | Test Set | mIoU w/o SegFix | mIoU w/ SegFix |
| :----: | :----: | :--: | :--: |
| HRNet-W48 | val | 81.1 | 81.6 |
| HRNet-W48 + OCR | test | 84.2 | 84.5 |

### Use SegFix in openseg

To apply SegFix, you should first down the offset files [offset_instance.zip](https://drive.google.com/open?id=1iDP2scYmy51XJww-888oouNpRBksmrkv) to `$DATA_ROOT/cityscapes`, and then extract the archive.

Several scripts in openseg have built-in support for SegFix post-processing. For example, you can first run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to get the baseline prediction of HRNet-W48 + OCR model, then run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh segfix 1 val` to further apply SegFix on the prediction.

### Use SegFix for your own label files

You can use `scripts/cityscapes/segfix.py` to apply SegFix on your own label files. Usage:
```bash
python scripts/cityscapes/segfix.py \
  --input <path/to/your/label/dir> \
  --split <SPLIT> \
  [ --offset <OFFSET_DIR>] \
  [ --out <OUT_DIR>]
```
where 
  + `<SPLIT>` is `test` or `val`.
  + `<OFFSET_DIR>` is the location of SegFix offsets, default to `$DATA_ROOT/cityscapes/val/offset_pred/semantic/offset_hrnext/` or `$DATA_ROOT/cityscapes/test_offset/semantic/offset_hrnext/`.
  + `<OUT_DIR>` is an optional output directory.


## SegFix for Instance Segmentation

### Released prediction files

We have released the prediction of some state-of-the-arts approaches and their SegFixed results. The files can be found [here](https://drive.google.com/open?id=184RXq8-RT8cdt5ojQGa1ziGN5iOxU1Xh). The performances are listed in the table below:

| Method | Test Set | AP w/o SegFix | AP w/ SegFix |
| :----: | :----: | :--: | :--: |
| MaskRCNN (w/ COCO, detectron2) | val | 36.5 | 38.2 |
| PointRend (w/ COCO, detectron2) | val | 37.9 | 39.5 |
| MaskRCNN (w/ COCO, detectron2) | test | 32.0 | 33.3 |
| PointRend (w/ COCO, detectron2) | test | 33.3 | 34.8 |

### Use SegFix for your own label files

You can use `scripts/cityscapes/segfix_instance.py` to apply SegFix on your own label files. Usage:
```bash
python scripts/cityscapes/segfix.py \
  --input <path/to/your/label/dir> \
  --split <SPLIT> \
  [ --offset <OFFSET_DIR>] \
  [ --out <OUT_DIR>]
```
where 
  + `<SPLIT>` is `test` or `val`.
  + `<OFFSET_DIR>` is the location of SegFix offsets, default to `$DATA_ROOT/cityscapes/val/offset_pred/semantic/offset_hrnext/` or `$DATA_ROOT/cityscapes/test_offset/semantic/offset_hrnext/`.
  + `<OUT_DIR>` is an optional output directory.


