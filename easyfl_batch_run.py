# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time       : 2023/7/11 5:04 下午
# @Author     : Zipeng Ye
# @Affliction : Harbin Institute of Technology, Shenzhen
# @File       : easyfl_batch_run.py

import os
import easyfl
import argparse
from Custom_FL import CustomizedServer, CustomizedClient, ResNet18_cifar100, ResNet18_tinyImageNet, \
    get_FedTinyImageNet, get_FedEMNIST, ResNet18_EMNIST, ResNet18_cifar10


def running(dataset='cifar10', num_of_clients=30, split_type='dir', rounds=100, clients_per_round=6,
            local_epoch=5, lr=0.1, method='default', data_augment_strategy='default', alpha=0.5):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f'rank:{rank} | local_rank:{local_rank} | world_size:{world_size}')

    config = {
        "data": {
            "dataset": dataset,
            "num_of_clients": num_of_clients,
            "split_type": split_type,
            "class_per_client": 1,  # Only applicable when the split_type is 'class'.
            "alpha": alpha,
            "min_size": 10,
            "root": './data/',
        },
        "server": {
            "rounds": rounds,
            "clients_per_round": clients_per_round,
            "aggregation_strategy": 'FedAvg',
            "save_model_every": 50
        },
        "client": {"local_epoch": local_epoch,
                   "optimizer": {
                       "type": "SGD",
                       "lr": lr,
                       "step_lr": True,
                       "weight_decay": 5e-4,
                       "momentum": 0.9,
                   },
                   "training_strategy": {
                       "method": method, #'scaffold', "FedProx",  # "default"
                       "weight": 0.001,  # for "FedProx"
                   },
                   "data_augment_strategy": data_augment_strategy,
                   "data_augment_weight": 1e-3,
                   "rounds": rounds,
                   },
        "model": "resnet18",
        "test_mode": "test_in_server",
        "is_remote": None,
        "local_port": None,
        "gpu": world_size,
        "distributed": {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "init_method": 'env://',
        },
    }
    if config['data']['dataset'] == 'cifar10':
        model = ResNet18_cifar10()
        easyfl.register_model(model)
    elif config['data']['dataset'] == 'cifar100':
        model = ResNet18_cifar100()
        easyfl.register_model(model)
    elif config['data']['dataset'] == 'tiny-image-net':
        model = ResNet18_tinyImageNet()
        root = config['data']['root']
        train_data, test_data = get_FedTinyImageNet(root,
                                                    num_of_clients=config['data']['num_of_clients'],
                                                    split_type=config['data']['split_type'],
                                                    alpha=config['data']['alpha'],
                                                    min_size=config['data']['min_size'],
                                                    class_per_client=config['data']['class_per_client'])
        easyfl.register_dataset(train_data, test_data)
        easyfl.register_model(model)
    elif config['data']['dataset'] == 'emnist':
        model = ResNet18_EMNIST()
        root = config['data']['root']
        train_data, test_data = get_FedEMNIST(root,
                                              num_of_clients=config['data']['num_of_clients'],
                                              split_type=config['data']['split_type'],
                                              alpha=config['data']['alpha'],
                                              min_size=config['data']['min_size'],
                                              class_per_client=config['data']['class_per_client'])
        easyfl.register_dataset(train_data, test_data)
        easyfl.register_model(model)

    easyfl.register_server(CustomizedServer)
    easyfl.register_client(CustomizedClient)
    easyfl.init(config)
    easyfl.run()


if __name__ == '__main__':
    parser0 = argparse.ArgumentParser(description='none')
    parser0.add_argument('--mode', type=int, default=0, help='choose function to run')
    args = parser0.parse_args()
    if args.mode == 0:
        running(dataset='emnist', num_of_clients=100, split_type='dir', rounds=100, clients_per_round=20,
                local_epoch=2, lr=0.1, method='default', data_augment_strategy='default')
    elif args.mode == 1:
        running(dataset='cifar10', num_of_clients=100, split_type='dir', rounds=300, clients_per_round=20,
                local_epoch=5, lr=0.1, method='FedProx', data_augment_strategy='default')
    elif args.mode == 2:
        running(dataset='cifar100', num_of_clients=100, split_type='dir', rounds=500, clients_per_round=20,
                local_epoch=5, lr=0.1, method='scaffold', data_augment_strategy='mix')
    elif args.mode == 3:
        running(dataset='tiny-image-net', num_of_clients=30, split_type='dir', rounds=500, clients_per_round=6,
                local_epoch=1, lr=0.2, method='default', data_augment_strategy='default')
