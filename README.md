# Use different GhostModules to Accelarate Neural Networks  
The group project of  CMSC5743-Efficient Computing of Deep Neural Networks  2020fall

## Introduction
This repository contains PyTorch implementation of：
- [GhostModule,GhostBlock]() 
- [New Ghost Module](https://github.com/dnnYANGZHANG/dnn) we design in this course
- VGGnet,ResNet

We will compare the following networks:
- 1.Original networks
- 2.Networks with GhostModule
- 3.Networks with New Ghost Module we design

We will do the following comparations between the belowing 3 networks :
- 1.Compare the size of Trainnable parameters
- 2.Compare the speed on GPU
- 3.Compare the accuracy
- 4.Compare the speed on Mobile devices without GPU（Can be regarded as on CPU)


## Table of Contents

- [Team Members](#team-members)
- [Usage](#usage)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Enviroment](#enviroment)
- [Brief Result](#brief-result)
- [Citation](#citation)

## Team Members

> ​	[@Zhang Hongquan](https://github.com/horcham)  1155148260
> 
> ​	[@Yang Zekun](https://github.com/Dopeeee)      1155145496
> 


## Usage
### How to Train from Scratch:
```sh
$ python train.py 
```
### How to train from checkpoint
```sh
$ 待补充
```
### How to evaluate 
```sh
$ 待补充
```

Note: 

## Dataset
We use [Cifar10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset in this course project

## Requirements
- python 3.6
- Nvidia GPU + CUDA cuDNN
- Pytorch

## Enviroment
- Nvidia 2070

## Brief Result
- 1.The size of Trainnable parameters
结果1
- 2.The speed on GPU
结果2
- 3.The accuracy
结果3
- 4.The speed on Mobile devices without GPU（Can be regarded as on CPU)
结果4

## Citation
  



