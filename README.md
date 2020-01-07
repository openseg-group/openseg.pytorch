# openseg.pytorch


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-coco-stuff-test)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=object-contextual-representations-for)

	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-lip-val)](https://paperswithcode.com/sota/semantic-segmentation-on-lip-val?p=object-contextual-representations-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/object-contextual-representations-for/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=object-contextual-representations-for)

## News

- 2020/01/07
"HRNet+OCR[Mapillary+Coarse]" currently achieves **84.26%** on Cityscapes test with better Mapillary pretraining, where we pretrain the HRNet+OCR model on the original Mapillary train and achieve **50.8%** on Mapillary val. We believe our approach can futher benefit from more expensive training settings as in the recent Panoptic-DeepLab. We can expect higher performance with our novel post-processing mechanism.

- 2020/01/03
"HRNet+OCR" will be made open-source in the code-base [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) very soon, thanks for your patience.

- 2020/01/02
Please email us (yuyua@microsoft.com) if you need the code for our OCR module and we would like to share it with you ASAP. We also hope you could try our method in your own code base and share the results with us.

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



## Acknowledgment
This project is developed based on the [segbox.pytorch](https://github.com/donnyyou/segbox.pytorch) and the author of segbox.pytorch donnyyou retains all the copyright of the reproduced Deeplabv3, PSPNet related code. 
