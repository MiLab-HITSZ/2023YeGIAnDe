# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time       : 2023/7/11 5:04 下午
# @Author     : Zipeng Ye
# @Affliction : Harbin Institute of Technology, Shenzhen
# @File       : gen_dataset_pre.py

from Custom_FL import get_FedTinyImageNet
from Custom_FL import get_FedEMNIST

root = './data/'
# train, test = get_FedTinyImageNet(root, 100, 'dir', 0.1, 10, 1)
train, test = get_FedEMNIST(root, 100, 'dir', 0.5, 10, 1)
