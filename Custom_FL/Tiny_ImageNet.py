# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time       : 2023/7/11 5:04 下午
# @Author     : Zipeng Ye
# @Affliction : Harbin Institute of Technology, Shenzhen
# @File       : Tiny_ImageNet.py

import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from easyfl.datasets.dataset import FederatedTensorDataset
from .data_distribute import data_simulation
import torchvision.transforms as transforms
from .data_download import TinyImageNet


def get_raw_TinyImageNet(root):
    path = root + 'tiny-imagenet-200/raw_data_dic.npy'
    if not os.path.exists(path):
        train_dataset = TinyImageNet(
            root=root, split="train", download=True, transform=None, cached=True,
        )
        val_dataset = TinyImageNet(
            root=root, split="val", download=True, transform=None, cached=True,
        )
        dic_TinyImageNet = {}
        train_dic = {}
        train_dic['x'] = np.array(train_dataset.cache)
        train_dic['y'] = train_dataset.targets
        test_dic = {}
        test_dic['x'] = np.array(val_dataset.cache)
        test_dic['y'] = val_dataset.targets
        del train_dataset, val_dataset
        dic_TinyImageNet['train'] = train_dic
        dic_TinyImageNet['test'] = test_dic
        np.save(path, dic_TinyImageNet)
    else:
        dic_TinyImageNet = np.load(path, allow_pickle=True).item()

    train_dic = dic_TinyImageNet['train']
    test_data = dic_TinyImageNet['test']
    return train_dic, test_data


def get_FedTinyImageNet(root, num_of_clients=10, split_type='iid', alpha=0.5, min_size=10,
                        class_per_client=1, stack_x=True, quantity_weights=None):
    if root[-1] != '/':
        root += '/'
    dir = f'tin_{split_type}_{num_of_clients}_{min_size}_{class_per_client}_{alpha}_0/'
    all_dir = root + 'tiny-imagenet-200/' + dir
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4789886474609375, 0.4457630515098572, 0.3944724500179291),
                             std=(0.27698642015457153, 0.2690644860267639, 0.2820819020271301)),
    ])
    if not os.path.exists(all_dir):
        os.mkdir(all_dir)
        train_dic, test_data = get_raw_TinyImageNet(root)
        clients, train_data = data_simulation(train_dic['x'], train_dic['y'], num_of_clients, split_type, weights=quantity_weights,
                                                              alpha=alpha, min_size=min_size, class_per_client=class_per_client, stack_x=stack_x)
        np.save(all_dir+'train.npy', train_data)
        np.save(all_dir+'test.npy', test_data)
        train_data = FederatedTensorDataset(train_data,
                                            simulated=True,
                                            do_simulate=False,
                                            process_x=None,
                                            process_y=None,
                                            transform=transform)
        test_data = FederatedTensorDataset(test_data,
                                           simulated=False,
                                           do_simulate=False,
                                           process_x=None,
                                           process_y=None,
                                           transform=transform)
        return train_data, test_data
    else:
        train_data = np.load(all_dir+'train.npy', allow_pickle=True).item()
        test_data = np.load(all_dir+'test.npy', allow_pickle=True).item()
        train_data = FederatedTensorDataset(train_data,
                                            simulated=True,
                                            do_simulate=False,
                                            process_x=None,
                                            process_y=None,
                                            transform=transform)
        test_data = FederatedTensorDataset(test_data,
                                           simulated=False,
                                           do_simulate=False,
                                           process_x=None,
                                           process_y=None,
                                           transform=transform)
        return train_data, test_data

