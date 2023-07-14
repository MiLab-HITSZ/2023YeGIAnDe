# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/7/11 5:04 下午
# @Author      : Zipeng Ye
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : custom_dataset.py

import os
start = 0
end = 1
print(f'from {start} to {end-1}')
for i in range(start, end):
    command = f"torchrun --nproc_per_node=6 --master_port=1357 easyfl_batch_run.py --mode {int(i)}"
    os.system(command)
print('The training for all tasks is over!!!')
