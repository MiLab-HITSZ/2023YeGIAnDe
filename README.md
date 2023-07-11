# Gradient Inversion Attacks: A Comprehensive Analysis and Practical Defense Approach

Implementation of paper *"Z. Ye, W, Luo and Q. Zhou, et. al., Gradient Inversion Attacks: A Comprehensive Analysis and Practical Defense Approach."*


# Requirements

I have tested on:

- PyTorch 1.13.0e
- CUDA 11.0


# Usage

### If you want to test the gradient inversion attacks:

> python main_attack_mix_defense_224img.py

### If you want to simulate the distributed learning by default settings (dataset: EMNIST, aggregation=FedAvg, distribution=Dir(0.5), #clients=100, #round=100): 

1. run `gen_dataset_pre.py` to get training data with the above settings.
2. run `easyfl_shell.py` to simulate the distributed learning with 100 clients.

### or simulating the distributed learning using customed settings:

- changing the parameters in `gen_dataset_pre.py`.
- changing the parameters in `easyfl_batch_run.py`.
- running `easyfl_shell.py`.
 
 # REFERENCES
 
 *https://github.com/rosinality/stylegan2-pytorch*
 
