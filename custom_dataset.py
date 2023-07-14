# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/7/11 5:04 下午
# @Author      : Zipeng Ye
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : custom_dataset.py


import torch
import hydra
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision.utils as utl
from mix_methods import mixup, mix_recombine, mix_channel, mix_reverse, mix_poscomp, double_mix
import math

class CustomData:
    def __init__(self, data_dir, dataset_name, case, mix=None, only_mix=True, noise=0):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.case = case
        self.mix = mix
        self.om = only_mix
        self.noise = noise
        self.extract_mean_std()

    def get_data_cfg(self):
        with hydra.initialize(config_path='breaching/config/case/data', version_base='1.1'):
            cfg = hydra.compose(config_name=self.dataset_name)
        return cfg

    def get_case_cfg(self):
        with hydra.initialize(config_path='breaching/config/case', version_base='1.1'):
            cfg = hydra.compose(config_name=self.case)
        return cfg

    def extract_mean_std(self):
        cfg = self.get_data_cfg()
        self.mean = torch.as_tensor(cfg.mean)[None,:,None,None]
        self.std  = torch.as_tensor(cfg.std)[None,:,None,None]

    def process_data(self, sec_input4=False):
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
            ]
        )
        # trans = transforms.ToTensor()
        cfg_case = self.get_case_cfg()
        file_name_li = os.listdir(self.data_dir)
        file_name_list = sorted(file_name_li, key=lambda x: int(x.split('-')[0]))
        assert len(file_name_list) >= int(cfg_case.user.num_data_points)
        imgs = []
        labels_ = []
        # 打乱
        # random.seed(215)
        # random.shuffle(file_name_list)
        #####################################
        for file_name in file_name_list[0:int(cfg_case.user.num_data_points)]:
            img = Image.open(self.data_dir+file_name)
            imgs.append(trans(img)[None,:])
            label = int(file_name.split('-')[0])
            # label = 20
            labels_.append(label)
        imgs = torch.cat(imgs, 0)
        labels = torch.tensor(labels_)
        inputs = (imgs-self.mean)/self.std
        alpha  = 1-self.noise
        inputs = pow(alpha, 0.5)*inputs + pow(1-alpha, 0.5)*torch.randn_like(inputs)
        if self.mix:
            if inputs.shape[0] != 2:
                raise ValueError(f'too much images ({inputs.shape[0]}) are mixed!!!')
            inputs, labels = self._mix_2_images(inputs[0], inputs[1], labels[0], labels[1])
            return dict(inputs=inputs, labels=labels)
        if sec_input4:
            if inputs.shape[0] != 4:
                raise ValueError(f'input size is ({inputs.shape[0]}), require 4!!!')
            inputs, labels = self._mix_4_images(inputs, labels)
            return dict(inputs=inputs, labels=labels)
        return dict(inputs=inputs, labels=labels)

    def get_initial_from_img(self, path):
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
            ]
        )
        img = trans(Image.open(path))[None,:]
        return (img-self.mean)/self.std

    def save_recover(self, recover, original=None, save_pth='', sature=False):
        using_sqrt_row = False
        if original is not None:
            if isinstance(recover, dict):
                batch = recover['data'].shape[0]
                recover_imgs = torch.clamp((recover['data'].cpu()*self.std+self.mean), 0, 1)
                if sature:
                    recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
                origina_imgs = torch.clamp((original['data'].cpu()*self.std+self.mean), 0, 1)
                all = torch.cat([recover_imgs, origina_imgs], 0)
                if using_sqrt_row:
                    utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
                else:
                    utl.save_image(all, save_pth, nrow=batch)
            else:
                batch = recover.shape[0]
                recover_imgs = torch.clamp((recover.cpu()*self.std+self.mean), 0, 1)
                if sature:
                    recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
                origina_imgs = torch.clamp((original['data'].cpu()*self.std+self.mean), 0, 1)
                all = torch.cat([recover_imgs, origina_imgs], 0)
                if using_sqrt_row:
                    utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
                else:
                    utl.save_image(all, save_pth, nrow=batch)
        else:
            if isinstance(recover, dict):
                batch = recover['data'].shape[0]
                recover_imgs = torch.clamp((recover['data'].cpu()*self.std+self.mean), 0, 1)
                if sature:
                    recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
                all = recover_imgs
                if using_sqrt_row:
                    utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
                else:
                    utl.save_image(all, save_pth, nrow=batch)
            else:
                batch = recover.shape[0]
                recover_imgs = torch.clamp((recover.cpu()*self.std+self.mean), 0, 1)
                if sature:
                    recover_imgs = transforms.ColorJitter(saturation=(sature, sature))(recover_imgs)
                all = recover_imgs
                if using_sqrt_row:
                    utl.save_image(all, save_pth, nrow=int(math.sqrt(batch)))
                else:
                    utl.save_image(all, save_pth, nrow=batch)

    def recover_to_0_1(self, recover):
        tmp = recover['data'].data.clone()
        trans = torch.clamp((tmp.cpu()*self.std+self.mean), 0, 1)
        return trans

    def pixel_0_1_to_norm(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0).clamp(0, 1)
        else: tensor = tensor.clamp(0, 1)
        return (tensor-self.mean)/self.std

    def _mix_2_images(self, img1, img2, label1, label2, alpha=0.5):
        nc = 1000
        if self.mix == 'double_mix':
            img1_, img2_, mix1, mix2, label_and_lambda = double_mix(img1, img2, label1, label2, alpha)
            if self.om:
                img_mixed = torch.cat([mix1[None,:], mix2[None,:]], dim=0)
                return img_mixed, label_and_lambda
            else:
                imgs = torch.cat([img1_[None,:], img2_[None,:], mix1[None,:], mix2[None,:]], dim=0)
                return imgs, label_and_lambda
        elif self.mix == 'mixup':
            img_mixed, label_and_lambda = mixup(img1, img2, label1, label2, alpha)
        elif self.mix == 'mix_recombine':
            img_mixed, label_and_lambda = mix_recombine(img1, img2, label1, label2, alpha)
        elif self.mix == 'mix_channel':
            img_mixed, label_and_lambda = mix_channel(img1, img2, label1, label2, alpha)
        elif self.mix == 'mix_reverse':
            img_mixed, label_and_lambda = mix_reverse(img1, img2, label1, label2, alpha)
        elif self.mix == 'mix_poscomp':
            img1, img2, img_mixed, label_and_lambda = mix_poscomp(img1, img2, label1, label2, alpha)
        else:
            raise ValueError(f'No mix method: {self.mix}')

        if self.om:
            return img_mixed.unsqueeze(0), label_and_lambda
        else:
            imgs = torch.cat([img1[None,:], img2[None,:], img_mixed[None,:]], dim=0)
            return imgs, label_and_lambda

    def _mix_4_images(self, inputs, labels):
        lam = 0.8
        mix_result = -lam * inputs
        tmp = inputs[[1,2,3,0]]
        mix_result = mix_result + (1-lam) * tmp
        new_inputs = torch.cat([inputs, mix_result], 0)
        new_labels = torch.cat([labels, labels], 0)
        return new_inputs, new_labels


if __name__ == '__main__':
    print('hello gmy')

