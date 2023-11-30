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

**Manuscript link:**  
  To be supplemented.
  
**Citation:**  
  To be supplemented.
  
**Description:**  
This repo contains the Pytorch implementation of 3D segmentation of BraTS 2019, BraTS 2020 with the proposed *Shape-Scale Co-Awareness Network*. 

## Methods
In this paper we propose a novel *Shape-Scale Co-Awareness Network* that integrates CNN, Transformer, and MLP to synchronously capture shape-aware features and scale-aware features to cope with the pattern-agnostic challenges in brain tumor image segmentation.  
### Network Framework
![network](https://github.com/jiangyu945/S2CA-Net/blob/c4f6b12edd45bc8e1a33e1d1883d6c1d611fd5e3/img/Framework.png)
### Enhanced Axial Shifted MLP
![EAS-MLP](https://github.com/jiangyu945/S2CA-Net/blob/08d955d0e9a89e5f0addf0aa19d7e86e6a4f26f1/img/EAS-MLP.png)

## Results
### Quantitative Results
![Comparison_BraTS_2019](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Comparison_BraTS2019.png)
![Comparison_BraTS_2020](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Comparison_BraTS2020.png)
![Comparison_BraTS](https://github.com/jiangyu945/S2CA-Net/blob/ca185fef15421e18c2433b3f25c860e71eec05be/img/Comparison.png)
### Qualitative Results
![vis3d](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Visualization_Comparison.png)
### Ablation Analysis
![Ablation_component](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Ablation_component.png)
![Ablation_model_scale](https://github.com/jiangyu945/S2CA-Net/blob/c257a2c983c4852fa26a585e667a282690c2a61d/img/Ablation_model_scale.png)
## Usage
### Installation
Install the necessary python packages as defined in environment.yaml. We recommend using conda. You can create the environment using
```shell
conda env create -f environment.yml
```
If you run into problems, you can try using different versions of these packages.

### Data Preparation
Please download BraTS 2019, BraTS 2020 data according to https://www.med.upenn.edu/cbica/brats2019/registration.html and https://www.med.upenn.edu/cbica/brats2020/registration.html.  
Unzip downloaded data at `./dataset` folder (please create one) and remove all the csv files in the folder, or it will cause errors.
The implementation assumes that the data is stored in a directory structure like  
- dataset
  - BraTS2019
    -  MICCAI_BraTS_2019_Data_Training_Merge
       - BraTS19_2013_0_1
         - BraTS19_2013_0_1_flair.nii.gz
         - BraTS19_2013_0_1_t1.nii.gz
         - BraTS19_2013_0_1_t1ce.nii.gz
         - BraTS19_2013_0_1_t2.nii.gz
         - BraTS19_2013_0_1_seg.nii.gz
       - BraTS19_2013_1_1
           - ... 
    -  MICCAI_BraTS_2019_Data_Validation
       - BraTS19_CBICA_AAM_1
         - BraTS19_CBICA_AAM_1_flair.nii.gz
         - BraTS19_CBICA_AAM_1_t1.nii.gz
         - BraTS19_CBICA_AAM_1_t1ce.nii.gz
         - BraTS19_CBICA_AAM_1_t2.nii.gz
       - BraTS19_CBICA_ABT_1
         - ...
  - BraTS2020
    - MICCAI_BraTS2020_TrainingData
      - ...
    - MICCAI_BraTS2020_ValidationData
      - ...

### Pretrained Checkpoint
We provide ckpt download via Google Drive or Baidu Netdisk. Please download the checkpoint from the url below:  
#### Google Drive
url: https://drive.google.com/drive/folders/1-7kAjsYoQWBGLjilZVkFkxNCuiHqkOl5?usp=sharing
#### Baidu Netdisk
url：https://pan.baidu.com/s/10FlPjgh8F2y7RSHFCqRofw?pwd=s67e
extraction code (提取码)：s67e  

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
### Reference
[PANet](https://github.com/hsiangyuzhao/PANet)  
[TransBTS](https://github.com/Rubics-Xuan/TransBTS)
