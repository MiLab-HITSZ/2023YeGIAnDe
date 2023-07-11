# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time       : 2023/7/11 5:04 下午
# @Author     : Zipeng Ye
# @Affliction : Harbin Institute of Technology, Shenzhen
# @File       : main_attack_mix_defense_cifar100.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


if __name__ == '__main__':

    cfg = breaching.get_config(overrides=["case=11_small_batch_cifar", "attack=our_vgg13_dyna_untrain"])

    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

    cfg.case.data.partition = 'balanced'
    cfg.case.user.user_idx  = 0
    cfg.case.model = "VGG13"

    cfg.case.user.provide_labels=False
    cfg.case.user.provide_buffers=False
    cfg.case.server.provide_public_buffers=False
    cfg.case.server.pretrained=False

    cfg.case.data.mix = None
    only_mix = True
    using_feature_mix = False  # set using_mix_defense=False when using_feature_mix=True
    using_mix_defense = False
    if using_mix_defense:
        cfg.case.data.mix = 'double_mix' # 'mix_poscomp' #'mix_recombine' #'mixup'
        only_mix = False
    secure_input = False

    apply_noise = False
    apply_prune = False

    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker_loss = torch.nn.CrossEntropyLoss()
    attacker = breaching.attacks.prepare_attack(server.model, attacker_loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    # if not cfg.case.server.pretrained:
    #     server.reconfigure_model("untrained")
    server_payload = server.distribute_payload()

    cus_data = CustomData(data_dir='custom_data/32_img/', dataset_name='CIFAR100', case='11_small_batch_cifar', mix=cfg.case.data.mix, only_mix=only_mix)
    shared_data, true_user_data = user.compute_local_updates(server_payload, secure_input=secure_input, apply_noise=apply_noise, apply_prune=apply_prune)

    true_pat = 'custom_data/recons/a_truth.jpg' if "save_dir" not in cfg.attack.keys() else cfg.attack.save_dir + 'a_truth.jpg'
    cus_data.save_recover(true_user_data, save_pth=true_pat)
    if 'labels' in true_user_data.keys():
        if not isinstance(true_user_data['labels'], dict):
            true_lab = true_user_data['labels'].cpu().tolist()
        else:
            true_lab = true_user_data['labels']['labels']
        print(f'------------True labels are {true_lab}--------------')

    # if using_mix_defense:
    #     shared_data['metadata']['num_data_points'] = 2
    # shared_data['metadata']['num_data_points'] = 4
    num_tmp = shared_data['metadata']['num_data_points']

    initial_data = None
    # initial_data1 = cus_data.get_initial_from_img('goldfish2.jpg')
    # initial_data2 = torch.randn(1,3,224,224)
    # initial_data = [torch.cat([initial_data1, initial_data2], dim=0)]

    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun, custom=cus_data, initial_data=initial_data)
    recon_path__ = 'custom_data/recons/final_rec.jpg' if "save_dir" not in cfg.attack.keys() else cfg.attack.save_dir + 'final_rec.jpg'
    cus_data.save_recover(reconstructed_user_data, true_user_data, recon_path__)
    tpath = cfg.attack.save_dir + 'recon.pt'
    torch.save(reconstructed_user_data, tpath)

    true_user_data["data"] = true_user_data["data"][0:num_tmp]
    metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
                                        server.model, order_batch=True, compute_full_iip=False,
                                        cfg_case=cfg.case, setup=setup)
    report_path = cfg.attack.save_dir + 'report.pt'
    torch.save(metrics, report_path)





