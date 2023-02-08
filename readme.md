# dataset_conv

This program is designed to convert `VOC2012`, `Cityscapes`, `ADE20K` or `COCO` datasets into anchor & non-anchor pairs for contrastive learning.

|anchor|non-anchor|
|:-:|:-:|
|![plane](/img4readme/2007_003778_anchor0.jpg)|![non-plane](/img4readme/2007_003778_Nanchor0.jpg)|

## Prerequisites

- C++ compiler that fully supports [C++ 20 feature of `ranges`](https://en.cppreference.com/w/cpp/20)
- OpenCV

## Prepare datasets

Download `VOC2012` (together with `SegmentationClassAug` from [Semantic Boundaries Dataset and Benchmark](http://home.bharathh.info/pubs/codes/SBD/download.html). Check 2 and 3 [here](https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md)), `Cityscapes`, `ADE20K` or `COCO` as you need.

### [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)

Download and extract it to wherever you want. Its directory structure should be the same with below.

```bash
$ tree /path/to/VOCdevkit/VOC2012 -d
├── Annotations
├── ImageSets
│   ├── Action
│   ├── Layout
│   ├── Main
│   ├── Segmentation
│   └── SegmentationAug # SBD
│       ├── test.txt
│       ├── train_aug.txt
│       ├── train.txt
│       ├── trainval_aug.txt
│       ├── trainval.txt
│       └── val.txt
├── JPEGImages
├── SegmentationClass
├── SegmentationClassAug # SBD
└── SegmentationObject
```

### [Cityscapes](https://www.cityscapes-dataset.com/downloads/)

Download and extract `leftImg8bit_trainvaltest.zip` (raw images) and `gtFine_trainvaltest.zip` (labels).

```bash
$ tree /path/to/cityscapes -d -L 2
├── gtFine
│   ├── test
│   ├── train
│   └── val
└── leftImg8bit
    ├── test
    ├── train
    └── val
```

### [COCO](https://cocodataset.org/#download)

Download and extract training set [train2017.zip](http://images.cocodataset.org/zips/train2017.zip) and corresponding segmentation label [stuffthingmaps_trainval2017.zip](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) to the same folder. The latter is a set of gray scale segmentation annotations provided in [cocostuff](https://github.com/nightrome/cocostuff#downloads).

```bash
$ tree /path/to/coco -d
├── stuffthingmaps_trainval2017
│   ├── train2017
│   └── val2017
└── train2017

4 directories
```

### [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/index.html#Download)

Download the full ADE20K dataset(you may need an account for that). Extract it to your desired path. Notice that `/ADE20K_2021_17_01` seems to be a date-based folder name. Please change relative path in `dataset_conv.cpp` in case maintainer of ADE20K update this dataset.

```bash
$ tree /path/to/ADE20K_2021_17_01 -d -L 3
└── images
    └── ADE
        ├── training
        └── validation
```

## Usage

Just simply give it your `VOC2012`, `Cityscapes`, `ade20k` or `coco` dataset path.

```bash
./dataset_conv --voc12 [path/to/VOCdevkit contains `VOC2012`] --aug --coco [/path/to/coco] --ade [/path/to/ADE20K_2021_17_01] --city [/path/to/cityscapes contains `gtFine` and `leftImg8bit`] --output_dir [desired output directory (default to current dir)]
```

Outputs will be written to `ContrastivePairs` in the path `--output_dir` points to.

```bash
$ tree /path/to/ContrastivePairs -L 1
├── ade20k
├── ADE_ImgList.txt
├── cityscapes
├── Cityscapes_ImgList.txt
├── coco
├── COCO_ImgList.txt
├── voc
└── VOC_ImgList.txt

4 directories, 4 files
```

Those `*_ImgList.txt` files will be read by `dataloader` in training program.
