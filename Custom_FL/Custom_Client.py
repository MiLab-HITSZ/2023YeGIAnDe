# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time       : 2023/7/11 5:04 下午
# @Author     : Zipeng Ye
# @Affliction : Harbin Institute of Technology, Shenzhen
# @File       : Custom_Client.py

import copy
from easyfl.client import BaseClient
import time
import logging
import torch
import math
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CustomizedClient(BaseClient):
    def __init__(self,
                 cid,
                 conf,
                 train_data,
                 test_data,
                 device,
                 **kwargs):
        super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.init_process_data_fn()
        self.confidence = 0.0
        self.local_c = None
        self.global_c = None
        self.global_model = None
        self.beta2 = 10
        self.count = 0
        print(f'training with strategy ----{self.conf.training_strategy.method}----')

    def init_process_data_fn(self):
        self.mix_feature_loss = self.get_0
        if self.conf.data_augment_strategy == "reverse":
            self.process_data = self._extend_by_reverse
        elif self.conf.data_augment_strategy == "reverse+fmix":
            self.process_data = self._extend_by_reverse
            self.mix_feature_loss = self.get_style_loss
        elif self.conf.data_augment_strategy == "mix":
            self.process_data = self._extend_by_mix
        elif self.conf.data_augment_strategy == "mix2":
            self.process_data = self._extend_by_mix2
        elif self.conf.data_augment_strategy == "mix+fmix":
            self.process_data = self._extend_by_mix
            self.mix_feature_loss = self.get_style_loss
        else:
            self.process_data = self._keep_data

    def prox_loss(self):
        # for FedProx
        loss = 0
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            loss += (w-w_t).norm(2)
        return loss * self.conf.training_strategy.weight/2

    def mix_feature_domain(self, x1, x2, lam, y):
        assert x1.dim() == 3
        mean1 = torch.mean(x1, dim=(1,2), keepdim=True)
        std1 = torch.std(x1, dim=(1,2), keepdim=True) + 1e-8
        mean2 = torch.mean(x2, dim=(1,2), keepdim=True)
        std2 = torch.std(x2, dim=(1,2), keepdim=True) + 1e-8
        # lam = 0.2
        # cx = (1-lam)*x1 + lam*x2
        # _mean1 = torch.mean(cx, dim=(1,2), keepdim=True)
        # _std1 = torch.std(cx, dim=(1,2), keepdim=True) + 1e-8
        _mean1 = (1-lam)*mean1 + lam*mean2
        _std1  = (1-lam)*std1 + lam*std2
        # _mean2 = (1-lam)*mean2 + lam*mean1
        # _std2  = (1-lam)*std2 + lam*std1
        new_x1 = ((x1-mean1)/std1*_std1 + _mean1).unsqueeze(0)
        # new_x2 = ((x2-mean2)/std2*_std2 + _mean2).unsqueeze(0)
        # tmp_y = [y,y]
        # return torch.cat([new_x1, new_x2], dim=0), tmp_y
        return new_x1, [y]

    def get_style_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn, device) -> torch.Tensor:
        # loss2 = self.get_conv2_loss(x, y, loss_fn, device)
        loss3 = self.get_conv3_loss(x, y, loss_fn, device)
        # loss4 = self.get_conv4_loss(x, y, loss_fn, device)
        loss = loss3  # + loss4 + loss3
        return loss * self.conf.data_augment_weight

    def get_0(self, *args):
        return 0.0

    def pre_train(self):
        if self.conf.training_strategy.method == "FedProx":
            self.global_model = copy.deepcopy(self.model).to(self.device)
        elif self.conf.training_strategy.method == "scaffold":
            self.global_model = copy.deepcopy(self.model).to(self.device)
        else:
            pass

    def train(self, conf, device='cpu'):
        if self.conf.training_strategy.method == "default":
            self.train_default(conf, device)
        elif self.conf.training_strategy.method == "FedProx":
            self.train_FedProx(conf, device)
        elif self.conf.training_strategy.method == "scaffold":
            self.train_scaffold(conf, device)
        else:
            raise ValueError(f'no training strategy named {self.conf.training_strategy.method}!!!')
        self.count += 1

    def train_default(self, conf, device='cpu'):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for batched_x, batched_y in self.train_loader:
                # -------------------------------------------------------------------
                batched_x, batched_y = self.process_data(batched_x, batched_y)
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y) + self.mix_feature_loss(x,y,loss_fn,device)
                # -------------------------------------------------------------------
                # bx1, bx2, y, y2, alpha = self.process_data(batched_x, batched_y)
                # bx1, bx2, y, y2 = bx1.to(device), bx2.to(device), y.to(device), y2.to(device)
                # optimizer.zero_grad()
                # out1 = self.model(bx1)
                # out2 = self.model(bx2)
                # loss = (loss_fn(out1, y) + 1.0*loss_fn(out2, y) + 0.0*loss_fn(out2, y2))/2
                # -------------------------------------------------------------------
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        # self.confidence = cal_mean_conf(out)  # calculate confidence by last prediction
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def train_FedProx(self, conf, device='cpu'):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for batched_x, batched_y in self.train_loader:
                batched_x, batched_y = self.process_data(batched_x, batched_y)
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y) + self.prox_loss() + self.mix_feature_loss(x,y,loss_fn,device)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        # self.confidence = cal_mean_conf(out)  # calculate confidence by last prediction
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))
        self.global_model = None  # 释放全局模型

    def train_scaffold(self, conf, device='cpu'):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        iter_times = 0
        for i in range(conf.local_epoch):
            batch_loss = []
            for batched_x, batched_y in self.train_loader:
                batched_x, batched_y = self.process_data(batched_x, batched_y)
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y) + self.mix_feature_loss(x,y,loss_fn,device)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                # ----------using scaffold------------
                iter_times += 1
                if self.local_c is not None:
                    self.update_model_by_scaffold()
                # ------------------------------------
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        # self.confidence = cal_mean_conf(out)  # calculate confidence by last prediction
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))
        # ----------using scaffold------------
        self.get_local_c_dci(iter_times)
        # ------------------------------------
        self.global_model = None  # 释放全局模型

    def download(self, model):
        """Download model from the server.

        Args:
            model (nn.Module): Global model distributed from the server.
        """
        # if self.compressed_model:
        #     self.compressed_model.load_state_dict(model.state_dict())
        # else:
        self.compressed_model = copy.deepcopy(model)

    def _extend_by_reverse(self, bx, by, **kwargs):
        # if not hasattr(self, 'rev_mask'):
        #     h, w = bx.shape[2], bx.shape[3]
        #     n_block = 3
        #     shape = (h, w)
        #     mask_shape = (int(h / n_block), int(w / n_block))
        #     self.mask, self.rev_mask = gen_mask(shape, mask_shape)
        # max_mask_weight = 1.0
        # weight = max_mask_weight * (1 - 2*self.confidence)
        new_bx1 = bx  # bx * (self.mask - self.rev_mask*weight)
        new_bx2 = -bx  # * (self.rev_mask - self.mask*weight)
        x = torch.cat([new_bx1, new_bx2], dim=0)
        y = torch.cat([by, by], dim=0)
        return x, y

    def _extend_by_mix(self, bx, by, shuffle=True, **kwargs):
        n = bx.shape[0]
        new_bx1 = bx
        new_bx2 = -1.0*bx
        shuffle = False
        idx = torch.arange(1,n+1)
        idx[-1] = 0
        kk = torch.randperm(n)
        tmp = new_bx1[kk] if shuffle else new_bx1[idx]
        alpha = 0.2
        new_bx2 = (1-alpha)*new_bx2 + alpha*tmp # + 0.2*torch.randn_like(new_bx2)
        x = torch.cat([new_bx1, new_bx2], dim=0)
        y = torch.cat([by, by], dim=0)
        return x, y

    def _extend_by_mix2(self, bx, by, shuffle=True, **kwargs):
        n = bx.shape[0]
        new_bx1 = bx
        new_bx2 = -1.0*bx
        if self.count < 20:
            shuffle = False
        else:
            shuffle = False
        idx = torch.arange(1,n+1)
        idx[-1] = 0
        kk = torch.randperm(n)
        tmp = new_bx1[kk] if shuffle else new_bx1[idx]
        y2 = by[kk] if shuffle else by[idx]
        alpha = 0.2
        new_bx2 = (1-alpha)*new_bx2 + alpha*tmp
        x = torch.cat([new_bx1, new_bx2], dim=0)
        y = torch.cat([by, by], dim=0)
        return new_bx1, new_bx2, by, y2, alpha

    def label_orient_combine(self, x, y, tmp, alpha):
        n = x.shape[0]
        m = torch.mode(y)[0]
        select_idx = (y==m).nonzero(as_tuple=True)[0]
        sn = len(select_idx)
        for i in range(n):
            tmpi = i % sn
            idx = select_idx[tmpi]
            x[i] = (1-alpha)*x[i] + alpha*tmp[idx] if y[i] == m else x[i]
        return x

    def _keep_data(self, bx, by, **kwargs):
        return bx, by

    def update_model_by_scaffold(self):
        with torch.no_grad():
            for name, para in self.model.named_parameters():
                para.data = para.data - self.conf.optimizer.lr * (self.global_c[name]-self.local_c[name])

    def get_local_c_dci(self, iter_nums):
        self.dci = {}
        for (p1, p2) in zip(self.global_model.named_parameters(), self.model.named_parameters()):
            self.dci[p1[0]] = (p1[1].data - p2[1].data)/iter_nums/self.conf.optimizer.lr
        if self.local_c is not None:
            for name in self.local_c.keys():
                self.dci[name] -= self.global_c[name]
                self.local_c[name] += self.dci[name]
        else:
            self.local_c = copy.deepcopy(self.dci)

    def get_conv2_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn, device):
        half = int(x.shape[0]/2)
        recon_x = []
        recon_y = []
        _conv3 = self.model.input_to_conv2(x)
        for i in range(half):
            assert y[i] == y[i+half]
            alpha = torch.rand(1)*10
            disb = torch.distributions.beta.Beta(alpha, self.beta2)
            lam = disb.sample((1,)).item()
            tmp, tmp_y = self.mix_feature_domain(_conv3[i], _conv3[i+half], lam, y[i])
            recon_x.append(tmp)
            recon_y += tmp_y
        new_conv3 = torch.cat(recon_x, dim=0)
        new_y = torch.tensor(recon_y).to(device)
        output = self.model.conv2_to_out(new_conv3)
        loss = loss_fn(output, new_y)
        return loss

    def get_conv3_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn, device):
        half = int(x.shape[0]/2)
        recon_x = []
        recon_y = []
        _conv3 = self.model.input_to_conv3(x)
        for i in range(half):
            assert y[i] == y[i+half]
            alpha = torch.rand(1)*10
            disb = torch.distributions.beta.Beta(alpha, self.beta2)
            lam = disb.sample((1,)).item()
            tmp, tmp_y = self.mix_feature_domain(_conv3[i], _conv3[i+half], lam, y[i])
            recon_x.append(tmp)
            recon_y += tmp_y
        new_conv3 = torch.cat(recon_x, dim=0)
        new_y = torch.tensor(recon_y).to(device)
        output = self.model.conv3_to_out(new_conv3)
        loss = loss_fn(output, new_y)
        return loss

    def get_conv4_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn, device):
        half = int(x.shape[0]/2)
        recon_x = []
        recon_y = []
        _conv3 = self.model.input_to_conv4(x)
        for i in range(half):
            assert y[i] == y[i+half]
            alpha = torch.rand(1)*10
            disb = torch.distributions.beta.Beta(alpha, self.beta2)
            lam = disb.sample((1,)).item()
            tmp, tmp_y = self.mix_feature_domain(_conv3[i], _conv3[i+half], lam, y[i])
            recon_x.append(tmp)
            recon_y += tmp_y
        new_conv3 = torch.cat(recon_x, dim=0)
        new_y = torch.tensor(recon_y).to(device)
        output = self.model.conv4_to_out(new_conv3)
        loss = loss_fn(output, new_y)
        return loss


def cal_mean_conf(out):
    return torch.mean(torch.max(F.softmax(out, dim=1), dim=1)[0])


def gen_mask(shape, mask_shape):
    mask = torch.ones(shape)
    rev_val = 0
    for i in range(0, int(math.ceil(shape[0] / mask_shape[0]))):
        for j in range(0, int(math.ceil(shape[1] / mask_shape[1]))):
            if (i + j) % 2 == 0:
                row_end = min((i + 1) * mask_shape[0], shape[0])
                col_end = min((j + 1) * mask_shape[1], shape[1])
                mask[i * mask_shape[0]:row_end, j * mask_shape[1]:col_end] = rev_val
    return mask, 1 - mask
