# openseg.pytorch

**Update on 2019/06/03.**

We achieve **83.3116%+** on the leaderboard of Cityscapes test set based on single model [HRNetV2](https://github.com/HRNet/HRNet-Semantic-Segmentation) + OCR.

**Update on 2019/05/27.**

We achieve SOTA on **6** different semantic segmentation benchmarks including: Cityscapes, ADE20K,  LIP, Pascal-Context, Pascal-VOC, COCO-Stuff. We provide the source code for Fast OCNet(OCR), Sparse OCNet(ISA), OCNet on all the six benchmarks. More benchmarks will be supported latter. We will consider release all the check-points and training log for the below experiments.

**82.0%+/83.3%+** on the test set of Cityscapes with only Train-Fine + Val-Fine datasets/Coarse datasets.

**45.5%+** on the val set of ADE20K.

**56.5%+** on the val set of LIP.

**56.0%+** on the val set of Pascal-Context.

**81.0%+** on the val set of Pascal-VOC with ss test. (DeepLabv3+ is 80.02% with only train-aug)

**40.5%+** on the val set of COCO-Stuff.


## Performances with openseg.pytorch

- Cityscapes (testing with single scale whole image)

| Methods | Backbone | Train.  mIOU | Val. mIOU | Test. mIOU (8W Iters) | BS | Iters | 
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|
| FCN-Stride8 | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | 84.21,84.23 | 75.96,75.85 | - | 8 | 4W | -
| [PSPNet]() | [3x3-Res101]() | 86.11,86.14 | 78.55,78.56 | - | 8 | 4W |
| [DeepLabV3]() | [3x3-Res101]() | 86.67,86.65 | 78.62,78,90 | - | 8 | 4W | 
| [BaseOCNet]() | [3x3-Res101]() | - | - | - | 8 | 4W |
| [BaseOCNet-v2]() | [3x3-Res101]() | 85.99 | 79.67 | - | 8 | 4W |
| [AspOCNet]() | [3x3-Res101]() | 86.30,86.32 | 79.53,79.60 | -| 8 | 4W |  
| [AspOCNet+OHEM+Val]() | [3x3-Res101]() | 88.75 | 88.67 | 81.63(ms+flip) | 8 | 8W |
| [FastBaseOCNet]()<br> | [3x3-Res101]() | 85.65,85.73 | 79.12,79.55 | - | 8 | 4W |
| [FastAspOCNet]() | [3x3-Res101]() | 86.32,86.47 | 79.59,79.61 | - | 8 | 4W |
| [FastAspOCNet+OHEM+Val]() | [3x3-Res101]() | 88.57 | 88.44 | 81.82(ms+flip) | 8 | 8W |
| [FastBaseOCNet+OHEM+Val]() | [3x3-Res101]() | 90.00 | 89.89 | 81.71(ms+flip) | 8 | 20W |
| [FastAspOCNet+OHEM+Val]() | [3x3-Res101]() | 90.94 | 90.64 | **82.06**(ms+flip) | 8 | 20W |

- ADE20K (testing with single scale whole image)

| Methods | Backbone  | Val. mIOU | PixelACC | BS | Iters |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [BaseOCNet]() | [3x3-Res50]() | - | - | 16 | 15W |
| [AspOCNet]() | [3x3-Res50]()  | 42.76,42.59 | 80.62,80.59 | 16 | 15W |
| [FastBaseOCNet]() | [3x3-Res50]() | - | - | 16 | 15W |
| [FastAspOCNet]() | [3x3-Res50]()  | 42.83,42.99 | 80.70,80.67 | 16 | 15W |
| [BaseOCNet]() | [3x3-Res101]()  | - | - | 16 | 15W |
| [AspOCNet]() | [3x3-Res101]()  | 44.10,44.27 | 81.34,81.44 | 16 | 15W |
| [FastBaseOCNet]() | [3x3-Res101]()  | 44.42 | 81.34 | 16 | 15W |
| [FastAspOCNet]() | [3x3-Res101]()  | 44.50,44.14 | 81.44,81.40 | 16 | 15W |
| [FastAspOCNet]() | [3x3-Res101]()  | **45.69**(ms+flip) | **82.04**(ms+flip) | 16 | 15W |

- LIP (testing with single scale whole image + left-right flip)

| Methods | Backbone  | Val. mIOU | PixelACC | BS | Iters |
|--------|:---------:|:------:|:------:|:------:|:------:|
| [CE2P+BaseOC]()     | [3x3-Res101]()  | 54.05 | 87.45 | 40 | 11W |
| [CE2P+ASPOC]()      | [3x3-Res101]()  | 54.72 | 87.73| 40 | 11W |
| [CE2P+FastBaseOC]() | [3x3-Res101]()  | 54.59 | 87.67 | 40 | 11W |
| [CE2P+FastAspOC]()  | [3x3-Res101]()  | **55.29** | **87.82** | 40 | 11W |


- Pascal-VOC (testing with single scale whole image)

- Pascal-Context (testing with single scale whole image)

- COCO-Stuff-10K (testing with single scale whole image)



## Acknowledgment
This project is developed based on the [segbox.pytorch](https://github.com/donnyyou/segbox.pytorch) and the author of segbox.pytorch donnyyou retains all the copyright of the reproduced Deeplabv3, PSPNet related code. 
