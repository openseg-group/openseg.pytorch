# openseg.pytorch

**News!!! @ 2019/07/23.**

We achieve **Rank#1** on the leaderboard of Cityscapes (with a single model) and the results will be public soon.


**Update @ 2019/06/19.**

We achieve **83.3116%+** on the leaderboard of Cityscapes test set based on single model [HRNetV2](https://github.com/HRNet/HRNet-Semantic-Segmentation) + OCR. [Cityscapes leaderboard](https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results)

We achieve **56.02%** on the leaderboard of ADE20K test set based on single model ResNet101 + OCR without any bells or whistles. [ADE20K leaderboard](http://sceneparsing.csail.mit.edu/eval/leaderboard.php)


**Update @ 2019/05/27.**

We achieve SOTA on **6** different semantic segmentation benchmarks including: **Cityscapes, ADE20K,  LIP, Pascal-Context, Pascal-VOC, COCO-Stuff**. We provide the source code for Fast OCNet(OCR), Sparse OCNet(ISA), OCNet on all the six benchmarks. More benchmarks will be supported latter. We will consider release all the check-points and training log for the below experiments.

**82.0%+/83.0%+** on the test set of Cityscapes with only Train-Fine + Val-Fine datasets/Coarse datasets.

**45.5%+** on the val set of ADE20K. 

**56.5%+** on the val set of LIP.

**56.0%+** on the val set of Pascal-Context.

**81.0%+** on the val set of Pascal-VOC with ss test. (DeepLabv3+ is 80.02% with only train-aug)

**40.5%+** on the val set of COCO-Stuff-10K.


## Performances with openseg.pytorch

- Cityscapes (testing with single scale whole image)

- ADE20K (testing with single scale whole image)

- LIP (testing with single scale whole image + left-right flip)

- Pascal-VOC (testing with single scale whole image)

- Pascal-Context (testing with single scale whole image)

- COCO-Stuff-10K (testing with single scale whole image)



## Acknowledgment
This project is developed based on the [segbox.pytorch](https://github.com/donnyyou/segbox.pytorch) and the author of segbox.pytorch donnyyou retains all the copyright of the reproduced Deeplabv3, PSPNet related code. 
