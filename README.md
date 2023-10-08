# Gradient Inversion Attacks: Impact Factors Analyses and Privacy Enhancement

Implementation of paper *"Z. Ye, W, Luo and Q. Zhou, et. al., Gradient Inversion Attacks: Impact Factors Analyses and Privacy Enhancement."*


# Requirements

I have tested on:

- PyTorch 1.13.0
- CUDA 11.0


# Usage

### If you want to test the gradient inversion attacks:

> python main_attack_mix_defense_224img.py

- note that the customized training data is put in directory `custom_data/1_img`

- or changing the parameters to test attacks with diffrent settings.

### If you want to simulate the distributed learning by default settings (dataset: EMNIST, aggregation=FedAvg, distribution=Dir(0.5), #clients=100, #round=100): 

1. run `gen_dataset_pre.py` to get training data with the above settings.
2. run `easyfl_shell.py` to simulate the distributed learning with 100 clients.

### or simulating the distributed learning using customized settings (with different aggregation method and defense method):

- changing the parameters in `gen_dataset_pre.py` (for generating diffrent data distrbution).
- changing the parameters in `easyfl_batch_run.py` (details see the script).
- running `easyfl_shell.py`.
 
 # REFERENCES
 
 *https://github.com/JonasGeiping/breaching*
 
 *https://github.com/EasyFL-AI/EasyFL*
 
