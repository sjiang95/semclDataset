# voc_conv

This program is designed to convert VOC2012 datasets into anchor & non-anchor pairs for contrastive learning.

|anchor|non-anchor|
|:-:|:-:|
|![plane](/img4readme/2007_000738_anchor0.jpg)|![non-plane](img4readme/2007_000738_Nanchor0.jpg)|

## usage

Just simply give it your `VOC2012` dataset path, which is expected to point to `/VOC2012`.

```cmd
./voc_conv --voc_path [VOC_root_path]
```

Outputs will be written to the path where you execute the program.
