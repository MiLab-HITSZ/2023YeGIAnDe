# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time       : 2023/7/11 5:04 下午
# @Author     : Zipeng Ye
# @Affliction : Harbin Institute of Technology, Shenzhen
# @File       : mix_methods.py

import torch
from torchvision import transforms
import math


def mixup(img1, img2, label1, label2, alpha):
    tmp = alpha*img1 + (1-alpha)*img2
    # tmp = img1+img2
    lab = {'labels': (label1, label2), 'lambda': (alpha, 1-alpha)}
    return tmp, lab


def mix_recombine(img1, img2, label1, label2, alpha):
    tmp = torch.zeros_like(img1)-2.0
    _,h,w = tmp.shape
    rh, rw = int(h/2), int(w/2)
    trans = transforms.Resize([rh, rw])
    img1_ = trans(img1)  * alpha
    img2_ = trans(img2)  * (1-alpha)
    tmp[:, 0:img1_.shape[1], 0:img1_.shape[2]] = img1_
    tmp[:, rh:rh+img2_.shape[1], rw:rw+img2_.shape[2]] = img2_
    lab = {'labels': (label1, label2), 'lambda': (alpha, 1-alpha)}
    return tmp, lab


def mix_channel(img1, img2, label1, label2, alpha):
    tmp = alpha*img1[[1,2,0]] + (1-alpha)*img2[[1,2,0]]
    lab = {'labels': (label1, label2), 'lambda': (alpha, 1-alpha)}
    return tmp, lab


def mix_reverse(img1, img2, label1, label2, alpha):
    tmp1 = -img1*0.5
    tmp2 = -img2*0.5
    # tmp = alpha*img1 + (1-alpha)*img2
    tmp = tmp1 + tmp2
    lab = {'labels': (label1, label2), 'lambda': (alpha, 1-alpha)}
    return tmp, lab


def mix_poscomp(img1, img2, label1, label2, alpha):
    # 原图和mix图位置互补
    mask_shape = [60,60]
    mask, rev_mask = gen_mask(img1[0].shape, mask_shape)
    img1_ = img1  # mask * img1 + rev_mask * img1
    img2_ = img2  # rev_mask * img2 - mask * img2
    # trans1 = transforms.ColorJitter(contrast=0.75)
    # img1_ = trans1(img1_)
    # img2_ = trans1(img2_)
    # trans = transforms.RandomAffine(degrees=90, translate=(0.2, 0.2))
    # trans = transforms.RandomVerticalFlip(p=1)
    # new_mask, new_rev_mask = gen_mask_move(mask, mask_shape)
    mix = torch.zeros_like(img1)
    mix = 1*(-img1-img2)
    # mix = -img1-img2  # rev_mask * img1 + mask * img1
    # mix = trans(mix)
    alpha = 0.5
    # mix = alpha*img1 + (1-alpha)*img2
    lab = {'labels': (label1, label2), 'lambda': (alpha, 1-alpha)}
    return img1_, img2_, mix, lab


def double_mix(img1, img2, label1, label2, alpha):
    # 原图和mix图位置互补
    mask_shape = [112,112]
    mask, rev_mask = gen_mask(img1[0].shape, mask_shape)
    alpha = 1
    img1_ = img1  # mask * img1 + rev_mask * img1
    img2_ = img2  # rev_mask * img2 - mask * img2
    mix1 = -img1 * 0.8 + img2 * 0.2
    mix2 = -img2 * 0.8 + img1 * 0.2
    # mix1 = alpha*img1 + (1-alpha)*img2
    # mix2 = alpha*img2 + (1-alpha)*img1
    # img1_ = mask * img1
    # img2_ = mask * img2
    # mix1  = rev_mask * img1
    # mix2  = rev_mask * img2
    lab = {'labels': (label1, label2), 'lambda': (alpha, 1-alpha)}
    return img1_, img2_, mix1, mix2, lab


def gen_mask(shape, mask_shape):
    mask = torch.ones(shape)*1
    rev_val = 0
    for i in range(0, int(math.ceil(shape[0]/mask_shape[0]))):
        for j in range(0, int(math.ceil(shape[1]/mask_shape[1]))):
            if (i+j) % 2 == 0:
                row_end = min((i+1)*mask_shape[0], shape[0])
                col_end = min((j+1)*mask_shape[1], shape[1])
                mask[i*mask_shape[0]:row_end, j*mask_shape[1]:col_end] = rev_val

    return mask, 1-mask  # -1*mask #+rev_val+1


def gen_mask_move(mask, mask_shape):
    rh, rw = mask_shape[0]//2, mask_shape[1]//2
    mh, mw = rh/mask.shape[0], rw/mask.shape[1]
    new_mask = transforms.RandomAffine(0, translate=(mh, mw), fillcolor=0)(mask.unsqueeze(0))[0]
    return new_mask, 1-new_mask
