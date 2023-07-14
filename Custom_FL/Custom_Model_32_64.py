# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/7/11 5:04 下午
# @Author      : Zipeng Ye
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : custom_dataset.py

from .ResNet import resnet18, resnet18_1channel

def ResNet18_cifar10():
    return resnet18(10)

def ResNet18_cifar100():
    return resnet18(100)

def ResNet18_tinyImageNet():
    return resnet18(200)

def ResNet18_EMNIST():
    return resnet18_1channel(47)