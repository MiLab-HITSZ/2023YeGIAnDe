# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/7/11 5:04 下午
# @Author      : Zipeng Ye
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : custom_dataset.py

import torchvision
import numpy as np
import os
from easyfl.datasets.dataset import FederatedTensorDataset
from .data_distribute import data_simulation
import torchvision.transforms as transforms

def get_raw_EMNIST(root):
    path = root + 'EMNIST/raw_data_dic.npy'
    if not os.path.exists(path):
        train_data = torchvision.datasets.EMNIST(root='../data', split='balanced', train=True,
                                                 download=True, transform=None)
        test_data = torchvision.datasets.EMNIST(root='../data', split='balanced', train=False,
                                                download=True, transform=None)
        dic_EMNIST = {}
        train_dic = {}
        train_dic['x'] = np.array(train_data.data[:,:,:,None])
        train_dic['y'] = train_data.targets.tolist()
        test_dic = {}
        test_dic['x'] = np.array(test_data.data[:,:,:,None])
        test_dic['y'] = test_data.targets.tolist()
        del train_data, test_data
        dic_EMNIST['train'] = train_dic
        dic_EMNIST['test'] = test_dic
        np.save(path, dic_EMNIST)
    else:
        dic_EMNIST = np.load(path, allow_pickle=True).item()

    train_dic = dic_EMNIST['train']
    test_data = dic_EMNIST['test']
    return train_dic, test_data


def get_FedEMNIST(root, num_of_clients=10, split_type='iid', alpha=0.5, min_size=10,
                        class_per_client=1, stack_x=True, quantity_weights=None):
    if root[-1] != '/':
        root += '/'
    dir = f'emnist_{split_type}_{num_of_clients}_{min_size}_{class_per_client}_{alpha}_0/'
    all_dir = root + 'EMNIST/' + dir
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.17510417103767395,),
                             std=(0.3332371413707733,)),
    ])
    if not os.path.exists(all_dir):
        os.mkdir(all_dir)
        train_dic, test_data = get_raw_EMNIST(root)
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