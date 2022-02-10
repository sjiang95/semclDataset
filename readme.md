# dataset_conv

This program is designed to convert `VOC2012`, `ADE20K` or `COCO` datasets into anchor & non-anchor pairs for contrastive learning.

|anchor|non-anchor|
|:-:|:-:|
|![plane](/img4readme/2007_000738_anchor0.jpg)|![non-plane](img4readme/2007_000738_Nanchor0.jpg)|

## Prepare datasets

Download `VOC2012`, `ADE20K` or `COCO` as you need.

### [voc2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)

Download and extract it to wherever you want. Its directory structure should be the same with below.

```bash
$ tree /path/to/VOCdevkit/VOC2012 -d
├── Annotations
├── ImageSets
│   ├── Action
│   ├── Layout
│   ├── Main
│   └── Segmentation
├── JPEGImages
├── SegmentationClass
└── SegmentationObject

9 directories
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

## usage

Just simply give it your `VOC2012`, `ade20k` or `coco` dataset path, which are expected to point to `/VOC2012`, `` and ``, respectively.

```bash
./dataset_conv --voc_path [path/to/VOCdevkit/VOC2012] --coco_path [/path/to/coco] --ade_path [/path/to/ADE20K_2021_17_01] --output_dir [desired output directory (default to current dir)]
```

Outputs will be written to the path where you execute the program.
