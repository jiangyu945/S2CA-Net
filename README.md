# Shape-Scale Co-Awareness Network for 3D Brain Tumor Segmentation
### :tada: Our work has been submitted to *IEEE Transactions on Medical Imaging*  
**Authors:**  
> Lifang Zhou[1][2][4]*, Yu Jiang[1], Weisheng Li[1], Jun Hu[3], Shenghai Zheng[1]

**Institution:**
> [1] School of Computer Science and Technology, Chongqing University of Posts and Telecommunications, Chongqing, China  
> [2] School of Software Engineering, Chongqing University of Posts and Telecommunications, Chongqing, China  
> [3] Department of Neurology, Southwest Hospital Third Military Medical University, Chongqing, China  
> [4] Key Laboratory of Advanced Manufacturing technology, Ministry of Education, Guizhou University, Guizhou, China
> *Corresponding Author: Lifang Zhou

manuscript link:  
-   

This repo contains the implementation of 3D segmentation of BraTS 2019, BraTS 2020 with the proposed *Shape-Scale Co-Awareness Network*.  
**If you use our code, please cite the paper:**  
> @ARTICLE{98521166,  
  author={Zhou, Lifang and Jiang, Yu and Li, Weisheng and Hu, Jun and Zheng, Shenghai},  
  journal={},   
  title={Shape-Scale Co-Awareness Network for 3D Brain Tumor Segmentation},   
  year={2024},  
  volume={},  
  number={},  
  pages={},  
  doi={}}  

## Methods
In this paper we propose a novel *Shape-Scale Co-Awareness Network* that integrates CNN, Transformer, and MLP to synchronously capture shape-aware features and scale-aware features to cope with the pattern-agnostic challenges in brain tumor image segmentation..  
### Network Framework
![network](https://github.com/jiangyu945/S2CA-Net/blob/c4f6b12edd45bc8e1a33e1d1883d6c1d611fd5e3/img/Framework.png)
### Local-Global Scale Mixer
![LGSM](https://github.com/jiangyu945/S2CA-Net/blob/971c0ac07c91ee0c1aab2a00ddc31d57d640937f/img/LGSM.png)
### Multi-level Context Aggregator
![MCA](https://github.com/jiangyu945/S2CA-Net/blob/0885948ed7f7042763b6ea28a2b2a21aef49cb86/img/MCA.png)
### Enhanced Axial Shifted MLP
![EAS-MLP](https://github.com/jiangyu945/S2CA-Net/blob/08d955d0e9a89e5f0addf0aa19d7e86e6a4f26f1/img/EAS-MLP.png)
### Multi-Scale Attentive Deformable Convolution
![MS-ADC](https://github.com/jiangyu945/S2CA-Net/blob/c00b72b1581d8adea11b6644ac98308ef843be6e/img/MS-ADC.png)

## Results
### Quantitative Results
![Comparison_BraTS_2019](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Comparison_BraTS2019.png)
![Comparison_BraTS_2020](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Comparison_BraTS2020.png)
### Qualitative Results
![vis3d](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Visualization_Comparison.png)
### Ablation Analysis
![Ablation_component](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Ablation_component.png)
![Ablation_model_scale](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Ablation_model_scale.png)

## Usage
### Data Preparation
Please download BraTS 2019, BraTS 2020 data according to https://www.med.upenn.edu/cbica/brats2019/data.html and https://www.med.upenn.edu/cbica/brats2020/data.html.  
Unzip downloaded data at `./dataset` folder (please create one) and remove all the csv files in the folder, or it will cause errors.
The implementation assumes that the data is stored in a directory structure like

./dataset
  - BraTS2019
    -  MICCAI_BraTS_2019_Data_Training_Merge
       - BraTS19_2013_0_1
         - BraTS19_2013_0_1_flair.nii.gz
         - BraTS19_2013_0_1_t1.nii.gz
         - BraTS19_2013_0_1_t1ce.nii.gz
         - BraTS19_2013_0_1_t2.nii.gz
         - BraTS19_2013_0_1_seg.nii.gz
       - BraTS19_2013_1_1
         - BraTS19_2013_1_1_flair.nii.gz
         - BraTS19_2013_1_1_t1.nii.gz
         - BraTS19_2013_1_1_t1ce.nii.gz
         - BraTS19_2013_1_1_t2.nii.gz
         - BraTS19_2013_1_1_seg.nii.gz
       ... 
       
    -  MICCAI_BraTS_2019_Data_Validation
       - BraTS19_CBICA_AAM_1
         - BraTS19_CBICA_AAM_1_flair.nii.gz
         - BraTS19_CBICA_AAM_1_t1.nii.gz
         - BraTS19_CBICA_AAM_1_t1ce.nii.gz
         - BraTS19_CBICA_AAM_1_t2.nii.gz
       - BraTS19_CBICA_ABT_1
       ...
         
  - BraTS2020
    - MICCAI_BraTS2020_TrainingData
    - MICCAI_BraTS2020_ValidationData

### Pretrained Checkpoint
We provide ckpt download via Google Drive or Baidu Netdisk. Please download the checkpoint from the url below:  
#### Google Drive
url: https://drive.google.com/file/d/1OwdKnM51rDvF3UiQDbcCWlPcYdc94-_O/view?usp=sharing
#### Baidu Netdisk
url：https://pan.baidu.com/s/14qM2k46mFnzT2RmI3sWcSw  
extraction code (提取码)：0512  

### Training
For default training configuration, we use patch-based training pipeline and use Adam optimizer. Deep supervision is utilized to facilitate convergence.
#### Train and validate on BraTS training set
```python
python train.py --model s2ca_net --patch_test --ds
```
#### Training on the entire BraTS training set
```python
python train.py --model s2ca_net --patch_test --ds --trainset
```
#### Breakpoint continuation for training
```python
python train.py --model s2ca_net --patch_test --ds -c CKPT
```
this will load the pretrained weights as well as the status of optimizer, scheduler and epoch.
#### PyTorch-native AMP training
```python
python train.py --model s2ca_net --patch_test --ds --mixed
```
if the training is too slow, please enable CUDNN benchmark by adding `--benchmark` but it will slightly affects the reproducibility.

### Inference
For default inference configuration, we use patch-based pipeline.
```python
python inference.py --model s2ca_net --patch_test --validation -c CKPT
```
### Inference with TTA
Inference with Test Time Augmentation(TTA).
```python
python inference.py --model s2ca_net --patch_test --validation -c CKPT --tta
```
