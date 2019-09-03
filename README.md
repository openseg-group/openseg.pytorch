# openseg.pytorch


## Updates

**Update @ 2019/08/09.**

We would like to support various backbones such as ResNet-101, WideResNet-38, HRNetV2-48.


**Update @ 2019/07/31.**

We have released the paper [ISA](https://arxiv.org/abs/1907.12273), which is very easy to use and implement while being much more efficient than OCNet or DANet based on conventional self-attention.

**Update @ 2019/07/23.**

We (HRNet + OCR w/ ASP) achieve **Rank#1** on the leaderboard of Cityscapes (with a single model) on 3 of 4 metrics.


**Update @ 2019/06/19.**

We achieve **83.3116%+** on the leaderboard of Cityscapes test set based on single model [HRNetV2](https://github.com/HRNet/HRNet-Semantic-Segmentation) + OCR. [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results)

We achieve **56.02%** on the leaderboard of ADE20K test set based on single model ResNet101 + OCR without any bells or whistles. [ADE20K leaderboard](http://sceneparsing.csail.mit.edu/eval/leaderboard.php)


**Update @ 2019/05/27.**

We achieve SOTA on **6** different semantic segmentation benchmarks including: **Cityscapes, ADE20K,  LIP, Pascal-Context, Pascal-VOC, COCO-Stuff**. We provide the source code for our approach on all the six benchmarks. More benchmarks will be supported latter. We will consider release all the check-points and training log for the below experiments.

**82.0%+/83.0%+** on the test set of Cityscapes with only Train-Fine + Val-Fine datasets/Coarse datasets.

**45.5%+** on the val set of ADE20K. 

**56.5%+** on the val set of LIP.

**56.0%+** on the val set of Pascal-Context.

**81.0%+** on the val set of Pascal-VOC with ss test. (DeepLabv3+ is 80.02% with only train-aug)

**40.5%+** on the val set of COCO-Stuff-10K.


## Performances with openseg.pytorch

- Cityscapes (testing with single scale whole image)

| Methods | Backbone | Train.  mIOU | Val. mIOU | Test. mIOU | BS | Iters | 
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|
| [FCN]() | [MobileNetV2]() | - | - | - | - | - |
| [FCN]() | [3x3-ResNet101]() | - | - | - | 8 | 4W |
| [FCN]() | [Wide-ResNet38]() | - | - | - | 8 | 4W |
| [FCN]() | [HRNetV2-48]() | - | - | - | 8 | 10W |
| [OCNet]() | [MobileNetV2]() | - | - | - | - | - |
| [OCNet]() | [3x3-ResNet101]() | - | - | - | 8 | 4W |
| [OCNet]() | [Wide-ResNet38]() | - | - | - | 16 | 2W |
| [OCNet]() | [HRNetV2-48]() | - | - | - | 8 | 10W |
| [ISA]() | [MobileNetV2]() | - | - | - | - | - |
| [ISA]() | [3x3-ResNet101]() | - | - | - | 8 | 4W |
| [ISA]() | [Wide-ResNet38]() | - | - | - | 16 | 2W |
| [ISA]() | [HRNetV2-48]() | - | - | - | 8 | 10W |
| [OCR]() | [MobileNetV2]() | - | - | - | - | - |
| [OCR]() | [3x3-ResNet101]() | - | - | - | 8 | 4W |  
| [OCR]() | [Wide-ResNet38]() | - | - | - | 16 | 2W | 
| [OCR]() | [HRNetV2-48]() | - | - | - | 8 | 10W |


- ADE20K (testing with single scale whole image)

| Methods | Backbone  | Val. mIOU | PixelACC | BS | Iters |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [FCN]() | [3x3-ResNet101]() | - | - | 16 | 15W |
| [FCN]() | [Wide-ResNet38]() | - | - | 16 | 15W |
| [FCN]() | [HRNetV2-48]() | - | - | 16 | 15W |
| [OCNet]() | [3x3-ResNet101]() | - | - |  16 | 15W |
| [OCNet]() | [Wide-ResNet38]() | - | - |  16 | 15W |
| [OCNet]() | [HRNetV2-48]() | - | - |  16 | 15W |
| [ISA]() | [3x3-ResNet101]() | - | - |  16 | 15W |
| [ISA]() | [Wide-ResNet38]() | - | - |  16 | 15W |
| [ISA]() | [HRNetV2-48]() | - | - |  16 | 15W |
| [OCR]() | [3x3-ResNet101]() | - | - |  16 | 15W |
| [OCR]() | [Wide-ResNet38]() | - | - |  16 | 15W |
| [OCR]() | [HRNetV2-48]() | - | - |  16 | 15W |

- LIP (testing with single scale whole image + left-right flip)

| Methods | Backbone  | Val. mIOU | PixelACC | BS | Iters |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [FCN]() | [3x3-ResNet101]() | - | - | 32 | 10W |
| [FCN]() | [Wide-ResNet38]() | - | - | 32 | 10W |
| [FCN]() | [HRNetV2-48]() | - | - | 32 | 10W |
| [OCNet]() | [3x3-ResNet101]() | - | - |  32 | 10W |
| [OCNet]() | [Wide-ResNet38]() | - | - |  32 | 10W |
| [OCNet]() | [HRNetV2-48]() | - | - |  32 | 10W |
| [ISA]() | [3x3-ResNet101]() | - | - |  32 | 10W |
| [ISA]() | [Wide-ResNet38]() | - | - |  32 | 10W |
| [ISA]() | [HRNetV2-48]() | - | - |  32 | 10W |
| [OCR]() | [3x3-ResNet101]() | - | - |  32 | 10W |
| [OCR]() | [Wide-ResNet38]() | - | - |  32 | 10W |
| [OCR]() | [HRNetV2-48]() | - | - |  32 | 10W |


- Pascal-VOC (testing with single scale whole image)

| Methods | Backbone  | Val. mIOU | PixelACC | BS | Iters |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [FCN]() | [3x3-ResNet101]() | - | - | 16 | 6W |
| [FCN]() | [Wide-ResNet38]() | - | - | 16 | 6W |
| [FCN]() | [HRNetV2-48]() | - | - | 16 | 6W |
| [OCNet]() | [3x3-ResNet101]() | - | - |  16 | 6W |
| [OCNet]() | [Wide-ResNet38]() | - | - |  16 | 6W |
| [OCNet]() | [HRNetV2-48]() | - | - |  16 | 6W |
| [ISA]() | [3x3-ResNet101]() | - | - |  16 | 6W |
| [ISA]() | [Wide-ResNet38]() | - | - |  16 | 6W |
| [ISA]() | [HRNetV2-48]() | - | - |  16 | 6W |
| [OCR]() | [3x3-ResNet101]() | - | - |  16 | 6W |
| [OCR]() | [Wide-ResNet38]() | - | - |  16 | 6W |
| [OCR]() | [HRNetV2-48]() | - | - |  16 | 6W |

- Pascal-Context (testing with single scale whole image)

| Methods | Backbone  | Val. mIOU | PixelACC | BS | Iters |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [FCN]() | [3x3-ResNet101]() | - | - | 16 | 3W |
| [FCN]() | [Wide-ResNet38]() | - | - | 16 | 3W |
| [FCN]() | [HRNetV2-48]() | - | - | 16 | 3W |
| [OCNet]() | [3x3-ResNet101]() | - | - |  16 | 3W |
| [OCNet]() | [Wide-ResNet38]() | - | - |  16 | 3W |
| [OCNet]() | [HRNetV2-48]() | - | - |  16 | 3W |
| [ISA]() | [3x3-ResNet101]() | - | - |  16 | 3W |
| [ISA]() | [Wide-ResNet38]() | - | - |  16 | 3W |
| [ISA]() | [HRNetV2-48]() | - | - |  16 | 3W |
| [OCR]() | [3x3-ResNet101]() | - | - |  16 | 3W |
| [OCR]() | [Wide-ResNet38]() | - | - |  16 | 3W |
| [OCR]() | [HRNetV2-48]() | - | - |  16 | 3W |

- COCO-Stuff-10K (testing with single scale whole image)

| Methods | Backbone  | Val. mIOU | PixelACC | BS | Iters |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [FCN]() | [3x3-ResNet101]() | - | - | 16 | 6W |
| [FCN]() | [Wide-ResNet38]() | - | - | 16 | 6W |
| [FCN]() | [HRNetV2-48]() | - | - | 16 | 6W |
| [OCNet]() | [3x3-ResNet101]() | - | - |  16 | 6W |
| [OCNet]() | [Wide-ResNet38]() | - | - |  16 | 6W |
| [OCNet]() | [HRNetV2-48]() | - | - |  16 | 6W |
| [ISA]() | [3x3-ResNet101]() | - | - |  16 | 6W |
| [ISA]() | [Wide-ResNet38]() | - | - |  16 | 6W |
| [ISA]() | [HRNetV2-48]() | - | - |  16 | 6W |
| [OCR]() | [3x3-ResNet101]() | - | - |  16 | 6W |
| [OCR]() | [Wide-ResNet38]() | - | - |  16 | 6W |
| [OCR]() | [HRNetV2-48]() | - | - |  16 | 6W |



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
```

## Acknowledgment
This project is developed based on the [segbox.pytorch](https://github.com/donnyyou/segbox.pytorch) and the author of segbox.pytorch donnyyou retains all the copyright of the reproduced Deeplabv3, PSPNet related code. 
