# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/7/11 5:04 下午
# @Author      : Zipeng Ye
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : custom_dataset.py

import copy

from easyfl.server import BaseServer
import torch.distributed as dist
import time
from easyfl.tracking import metric
from easyfl.protocol import codec


class CustomizedServer(BaseServer):
    # this server set barrier before training for each round
    def __init__(self, conf, **kwargs):
        super(CustomizedServer, self).__init__(conf, **kwargs)
        self.global_c = None

    def train(self):
        self.print_("--- start training ---")
        # if self.conf.is_distributed:
        #     dist.barrier()
        if self.conf.client.optimizer.step_lr:
            if self.conf.data.dataset == 'tiny-image-net':
                if self._current_round == int(self.conf.server.rounds/2) or \
                        self._current_round == int(self.conf.server.rounds)-5:
                    self.conf.client.optimizer.lr /= 10
                    print(f'learning rate is reset to {self.conf.client.optimizer.lr}')
            else:
                if self._current_round == int(self.conf.server.rounds/3) or \
                        self._current_round == int(self.conf.server.rounds*2/3):
                    self.conf.client.optimizer.lr /= 10
                    print(f'learning rate is reset to {self.conf.client.optimizer.lr}')
        self.selection(self._clients, self.conf.server.clients_per_round)
        self.grouping_for_distributed()
        self.compression()
        begin_train_time = time.time()
        self.distribution_to_train()
        self.aggregation()
        train_time = time.time() - begin_train_time
        self.print_("Server train time: {}".format(train_time))
        self.track(metric.TRAIN_TIME, train_time)

    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        uploaded_models = {}
        uploaded_weights = {}
        uploaded_metrics = []
        if self.conf.client.training_strategy.method == "scaffold":
            uploaded_c = {}
        for client in self.grouped_clients:
            # Update client config before training
            client.global_c = copy.deepcopy(self.global_c)
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self._current_round

            uploaded_request = client.run_train(self._compressed_model, self.conf.client)
            uploaded_content = uploaded_request.content

            model = self.decompression(codec.unmarshal(uploaded_content.data))
            uploaded_models[client.cid] = model
            uploaded_weights[client.cid] = uploaded_content.data_size
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

            client.model = None  # 释放空间
            client.compressed_model = None  # 释放空间
            if self.conf.client.training_strategy.method == "scaffold":
                uploaded_c[client.cid] = copy.deepcopy(client.dci)
                client.dci = {}  # 释放空间
                client.global_c = {}  # 释放空间

        self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)
        if self.conf.client.training_strategy.method == "scaffold":
            self._get_global_c(uploaded_c)

    def _get_global_c(self, uploaded_c):
        all_client = self.conf.data.num_of_clients
        tmp = uploaded_c.popitem()[1]
        for dci in uploaded_c.values():
            for name, para in dci.items():
                tmp[name] += para
        if self.conf.is_distributed:
            dist.barrier()
            for name in tmp.keys():
                dist.all_reduce(tmp[name], op=dist.ReduceOp.SUM)
        if self.global_c is None:
            for name in tmp.keys():
                tmp[name] /= all_client
            self.global_c = tmp
        else:
            for name in tmp.keys():
                self.global_c[name] += tmp[name]/all_client

