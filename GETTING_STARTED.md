
# Getting Started with openseg.pytorch

This document provides a brief intro of the usage of openseg.pytorch.

You can also refer to the https://github.com/openseg-group/openseg.pytorch/issues/14 for the guidelines on how to train the models on your own dataset based openseg.pytorch.


## Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch == 0.4.1
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.

You may use `pip install -r requirements.txt` to install the dependencies.

## Configuration

Before executing any scripts, your should first fill up the config file `config.profile` at project root directory. There are two items you should specify:

 + `PYTHON`, identifying your python executable.
 + `DATA_ROOT`, the root directory of your data. It should be the parent directory of `cityscapes`.


## Data Preparation

You need to download [Cityscapes](https://www.cityscapes-dataset.com/), [LIP](http://sysu-hcp.net/lip/), [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/) or [COCO-Stuff 10k v1.1](https://github.com/nightrome/cocostuff10k) datasets.

We arrange images and labels in another way. You could preprocess the files by running:

```bash
python lib/datasets/preprocess/cityscapes/cityscapes_generator.py --coarse True \
  --save_dir <path/to/preprocessed_cityscapes> --ori_root_dir <path/to/original_cityscapes>
python lib/datasets/preprocess/pascal_context/pascal_context_generator.py \
  --save_dir <path/to/preprocessed_context> --ori_root_dir <path/to/original_context>
python lib/datasets/preprocess/coco_stuff/coco_stuff_generator.py \
  --save_dir <path/to/preprocessed_cocostuff> --ori_root_dir <path/to/original_cocostuff>
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
├── coco_stuff_10k
│   ├── train
│   │   ├── image
│   │   └── label
│   ├── val
│   │   ├── image
│   │   └── label
```

## Inference with Pre-trained Models

Take HRNet-W48 + OCR on Cityscapes as an example. 

First you should refer to [MODEL_ZOO.md](MODEL_ZOO.md) to download its pre-trained weights `hrnet_w48_ocr_1_latest.pth`, and put it to `$PROJECT_ROOT/checkpoints/cityscapes/`. Then execute `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh val 1` to perform the inference. 

Here `1` is the id of experiments, and should match the id in the name `hrnet_w48_ocr_1_latest.pth`, which may be modified accordingly for other scripts.

After inference, you can retrieve the tested labelmaps and visualization at `$DATA_ROOT/cityscapes/hrnet_w48_ocr_1_latest_val/`.

## Training Models in openseg

First you should download ImageNet pre-trained weights from [here](https://drive.google.com/open?id=1ulZzlTulhIUvEa27joKLbas1TtbQOI7R), and put them under `$PROJECT_ROOT/pretrained_model/`. Then run `bash <script> train <name>` to start training, where `<script>` is path to the corresponding training script, and `<name>` is arbitary string to help you identify the experiments.

For example, to train HRNet-W48 + OCR on Cityscapes, you may run `bash scripts/cityscapes/hrnet/run_h_48_d_4_ocr.sh train 1`.

You can find path of all the scripts in [MODEL_ZOO.md](MODEL_ZOO.md).
