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
![network](https://github.com/jiangyu945/S2CA-Net/blob/ce264864b219320af10db127bf125d5b50bcdef0/img/Framework.png)
### Local-Global Scale Mixer
![LGSM](https://github.com/jiangyu945/S2CA-Net/blob/a99748cfbc13f40f8bec5f4bcd491d43d6451ee1/img/LGSM.png)
### Multi-level Context Aggregator
![MCA](https://github.com/jiangyu945/S2CA-Net/blob/a99748cfbc13f40f8bec5f4bcd491d43d6451ee1/img/MCA.png)
### Multi-Scale Attentive Deformable Convolution
![MS-ADC](https://github.com/jiangyu945/S2CA-Net/blob/a99748cfbc13f40f8bec5f4bcd491d43d6451ee1/img/MS-ADC.png)

## Results
### Quantitative Results
![Snipaste_2021-10-12_15-47-15](https://user-images.githubusercontent.com/53631393/136914282-3dd5a697-711b-4653-adb8-a6d2c98705f5.png)
### Qualitative Results
![vis3d](https://user-images.githubusercontent.com/53631393/136914543-023500b6-9a57-4f21-9f94-77961c7e9917.png)
### Ablation Analysis
![Snipaste_2021-10-12_15-47-32](https://user-images.githubusercontent.com/53631393/136914298-b76690c2-987d-4a3b-98da-9ab42f44ed10.png)

## Usage
### Data Preparation
Please download BraTS 2019, BraTS 2020 data according to `https://www.med.upenn.edu/cbica/brats2019/data.html` and `https://www.med.upenn.edu/cbica/brats2020/data.html`.  
Unzip downloaded data at `./dataset` folder (please create one) and remove all the csv files in the folder, or it will cause errors.

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
this will load the pretrained weights as well as the status of optimizer and scheduler.
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
