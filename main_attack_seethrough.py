# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time       : 2023/7/11 5:04 下午
# @Author     : Zipeng Ye
# @Affliction : Harbin Institute of Technology, Shenzhen
# @File       : main_attack_seethrough.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

try:
    import breaching
except ModuleNotFoundError:
    import os; os.chdir("..")
    import breaching

import torch
from custom_dataset import CustomData
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()
sys.path.append('./improved-diffusion/')


if __name__ == '__main__':

    cfg = breaching.get_config(overrides=["case=12_custom_imagenet", "attack=seethroughgradients_og_paper"])  # seethroughgradients_res34_batchsize8_dyna

    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    # print(setup)

    cfg.case.data.partition = 'unique-class' # 'balanced'
    cfg.case.user.user_idx  = 0
    cfg.case.model = "resnet34"

    cfg.case.user.provide_labels=False
    cfg.case.user.provide_buffers=False
    cfg.case.server.provide_public_buffers=True
    cfg.case.server.pretrained=True

    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    server_payload = server.distribute_payload()

    cus_data = CustomData(data_dir='custom_data/1_img/', dataset_name='ImageNet', case='12_custom_imagenet')
    shared_data, true_user_data = user.compute_local_updates(server_payload, custom_data=cus_data.process_data())
    ## comment part
    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun, custom=cus_data)
    recon_path__ = 'custom_data/recons/img_final.jpg' if "save_dir" not in cfg.attack.keys() else cfg.attack.save_dir + 'img_final.jpg'
    cus_data.save_recover(reconstructed_user_data, true_user_data, recon_path__)

    metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
                                    server.model, order_batch=True, compute_full_iip=False,
                                    cfg_case=cfg.case, setup=setup)
