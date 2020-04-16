
# Getting Started with openseg.pytorch

This document provides a brief intro of the usage of openseg.pytorch.


## Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
	You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- OpenCV, optional, needed by demo and visualization
- pycocotools: `pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`


## Configuration

Before executing any scripts, your should first fill up the config file `config.profile` at project root directory. There are two items you should specify:

 + `PYTHON`, identifying your python executable.
 + `DATA_ROOT`, the root directory of your data. It should be the parent directory of `cityscapes`.


## Data Preparation

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

## Inference Demo with Pre-trained Models

1. Pick a model and its config file from
	[model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md),
	for example, `mask_rcnn_R_50_FPN_3x.yaml`.
2. We provide `demo.py` that is able to run builtin standard models. Run it with:
```
cd demo/
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


## Training & Evaluation in Command Line

We provide a script in "tools/{,plain_}train_net.py", that is made to train
all the configs provided in detectron2.
You may want to use it as a reference to write your own training script for a new research.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:
```
cd tools/
./train_net.py --num-gpus 8 \
	--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```

The configs are made for 8-GPU training.
To train on 1 GPU, you may need to [change some parameters](https://arxiv.org/abs/1706.02677), e.g.:
```
./train_net.py \
	--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	--num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

For most models, CPU training is not supported.

To evaluate a model's performance, use
```
./train_net.py \
	--config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	--eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `./train_net.py -h`.
