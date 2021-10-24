# Relative Uncertainty Learning for Facial Expression Recognition
The official implementation of the following paper at NeurIPS2021:\
**Title:** Relative Uncertainty Learning for Facial Expression Recognition\
**Authors:** Yuhang Zhang, Chengrui Wang, Weihong Deng\
**Institute:** BUPT


## Abstract
In facial expression recognition (FER), the uncertainties introduced by inherent noises like ambiguous facial expressions and inconsistent labels raise concerns about the credibility of recognition results. To quantify these uncertainties and achieve good performance under noisy data, we regard uncertainty as a relative concept and propose an innovative uncertainty learning method called Relative Uncertainty Learning (RUL). Rather than assuming Gaussian uncertainty distributions for all datasets, RUL builds an extra branch to learn uncertainty from the relative difficulty of samples by feature mixup. Specifically, we use uncertainties as weights to mix facial features and design an add-up loss to encourage uncertainty learning. It is easy to implement and adds little or no extra computation overhead. Extensive experiments show that RUL outperforms state-of-the-art FER uncertainty learning methods in both real-world and synthetic noisy FER datasets. Besides, RUL also works well on other datasets such as CIFAR and Tiny ImageNet.

## Pipeline
![](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning/blob/main/imgs/overview_1.png)

## Feature Visualization
The feature distribution figure shows that RUL encourages intra-class compactness and inter-class seperability of the learned features. (0:Surprise, 1:Fear, 2:Disgust, 3:Happy, 4:Sad, 5:Angry, 6:Neutral)
![](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning/blob/main/imgs/feature_distribution.png)

## Train

**Torch** 

We train RUL with Torch 1.8.0 and torchvision 0.9.0.

**Dataset**

Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), put it into the dataset folder, and make sure that it has the same structure as bellow:
```key
- dataset/raf-basic/
         EmoLabel/
             list_patition_label.txt
         Image/aligned/
	     train_00001_aligned.jpg
             test_0001_aligned.jpg
             ...

```

**Pretrained backbone model**

Download the pretrained ResNet18 from [this](https://github.com/amirhfarzaneh/dacl) github repository, and then put it into the pretrained_model directory. We thank the authors for providing their pretrained ResNet model.

**Train the RUL model**

```key
cd src
python main.py --raf_path '../dataset/raf-basic' --label_path '../dataset/raf-basic/EmoLabel/list_patition_label.txt' --pretrained_backbone_path '../pretrained_model/resnet18_msceleb.pth'
```

**Accuracy**

![](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning/blob/main/imgs/accuracy.png)


