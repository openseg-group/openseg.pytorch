# openseg.pytorch


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-coco-stuff-test)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=object-contextual-representations-for)

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-lip-val)](https://paperswithcode.com/sota/semantic-segmentation-on-lip-val?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=object-contextual-representations-for)


## News

- 2019/12/06
We have released our code (with the training log and checkpoints) to provide a good baseline for the community. Thanks for your patience.


- 2019/11/19
We have updated the paper [OCR](https://arxiv.org/abs/1909.11065).
Our approach achieves **83.7%** and we can further achieve **84.0%** on Cityscapes test set with a novel yet simple model-agnostic post-processing scheme. Our model-agnostic post-processing scheme is a new work under progress, which can be applied to improve the results of any existing approaches without any re-training or fine-tuning.

- 2019/09/25
We have released the paper [OCR](https://arxiv.org/abs/1909.11065), which is method of our **Rank#2** entry to the leaderboard of Cityscapes.

- 2019/07/31
We have released the paper [ISA](https://arxiv.org/abs/1907.12273), which is very easy to use and implement while being much more efficient than OCNet or DANet based on conventional self-attention.

- 2019/07/23
We (HRNet + OCR w/ ASP) achieve **Rank#1** on the leaderboard of Cityscapes (with a single model) on 3 of 4 metrics.

- 2019/06/19
We achieve **83.3%+** on the leaderboard of Cityscapes test set based on single model [HRNetV2](https://github.com/HRNet/HRNet-Semantic-Segmentation) + OCR. [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results)
We achieve **56.02%** on the leaderboard of ADE20K test set based on single model ResNet101 + OCR without any bells or whistles. [ADE20K leaderboard](http://sceneparsing.csail.mit.edu/eval/leaderboard.php)

- 2019/05/27
We achieve SOTA on **6** different semantic segmentation benchmarks including: **Cityscapes, ADE20K,  LIP, Pascal-Context, Pascal-VOC, COCO-Stuff**. We provide the source code for our approach on all the six benchmarks.


## Model Zoo and Baselines

We provide a set of baseline results and trained models available for download in the [openseg Model Zoo](MODEL_ZOO.md).

## SegFix for InstanceSegmentation (TODO)

script at `scripts/cityscapes/segfix_instance.py`. Offset files at [offset_instance.zip](https://drive.google.com/open?id=1UXj6-XCXrPGAzDq3F1GGRpaF32nNTF4m).

## Citation
Please consider citing our work if you find it helps you,
```
@article{yuan2018ocnet,
  title={Ocnet: Object context network for scene parsing},
  author={Yuan Yuhui and Wang Jingdong},
  journal={arXiv preprint arXiv:1809.00916},
  year={2018}
}

@article{huang2019isa,
  title={Interlaced Sparse Self-Attention for Semantic Segmentation},
  author={Huang Lang and Yuan Yuhui and Guo Jianyuan and Zhang Chao and Chen Xilin and Wang Jingdong},
  journal={arXiv preprint arXiv:1907.12273},
  year={2019}
}

@article{yuan2019ocr,
  title={Object-Contextual Representations for Semantic Segmentation},
  author={Yuan Yuhui and Chen Xilin and Wang Jingdong},
  journal={arXiv preprint arXiv:1909.11065},
  year={2019}
}
```

## Acknowledgment
This project is developed based on the [segbox.pytorch](https://github.com/donnyyou/segbox.pytorch) and the author of segbox.pytorch donnyyou retains all the copyright of the reproduced Deeplabv3, PSPNet related code. 
