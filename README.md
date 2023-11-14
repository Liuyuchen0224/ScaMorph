# ScaMorph: A Robust Unsupervised Learning Model for Deformable Image Registration using Scale-aware Context Aggregation
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

**Keywords：Deep Learning， Deformable Image Registration， Convolutional Modulation，  Scale-aware Context Aggregation**

<img src="https://github.com/Liuyuchen0224/ScaMorph/blob/main/fig/Network.png" width="1000"/>

## Training and Testing
Training
```python
python train_ScaMorph.py --atlas_dir atlas_dir --train_dir train_dir --val_dir val_dir --gpu 0 
python train_ScaMorph_diff.py --atlas_dir atlas_dir --train_dir train_dir --val_dir val_dir --gpu 0 
```
Testing
```python
python infer.py
```

## Quantitative results

- IXI   
    <a href="https://github.com/Liuyuchen0224/ScaMorph/blob/main/log/IXI/ScaMorph_ncc1_reg1.csv">ScaMorph_ncc1_reg1.csv</a>    
    <a href="https://github.com/Liuyuchen0224/ScaMorph/blob/main/log/IXI/ScaMorphDiff_ncc1_reg1.csv">ScaMorphDiff_ncc1_reg1.csv</a>
- Brats   
    - T1toT2   
        <a href="https://github.com/Liuyuchen0224/ScaMorph/blob/main/log/Brats/T1toT2_ScaMorph_ncc1_reg1.csv">T1toT2_ScaMorph_ncc1_reg1.csv</a>   
        <a href="https://github.com/Liuyuchen0224/ScaMorph/blob/main/log/Brats/T1toT2_ScaMorphDiff_ncc1_reg1.csv">T1toT2_ScaMorphDiff_ncc1_reg1.csv</a>   
    - T2toT1   
        <a href="https://github.com/Liuyuchen0224/ScaMorph/blob/main/log/Brats/T2toT1_ScaMorph_ncc1_reg1.csv">T2toT1_ScaMorph_ncc1_reg1.csv</a>   
        <a href="https://github.com/Liuyuchen0224/ScaMorph/blob/main/log/Brats/T2toT1_ScaMorphDiff_ncc1_reg1.csv">T2toT1_ScaMorphDiff_ncc1_reg1.csv</a>
- Liver   
    <a href="https://github.com/Liuyuchen0224/ScaMorph/blob/main/log/Liver/ScaMorph_ncc1_reg1.csv">ScaMorph_ncc1_reg1.csv</a>   
    <a href="https://github.com/Liuyuchen0224/ScaMorph/blob/main/log/Liver/ScaMorphDiff_ncc1_reg1.csv">ScaMorphDiff_ncc1_reg1.csv</a>

## Reference:
<a href="https://github.com/voxelmorph/voxelmorph">VoxelMorph</a>
<a href="https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration">TransMorph</a>
<a href="https://github.com/xi-jia/LKU-Net">LKU-Net</a>
<a href="https://github.com/AFeng-x/SMT">SMT</a>