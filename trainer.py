#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainer.py
# @Author:
# @Date  : 2021/11/1 15:45
# @Desc  :
import copy
import time
import os

import torch
import numpy as np
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data_set import DataSet
from metrics import metrics_dict


class Trainer(object):

    def __init__(self, model, dataset: DataSet, args):
        self.model = model.to(args.device)
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.min_epoch = args.min_epoch
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        #self.train_writer = args.train_writer
        #self.test_writer = args.test_writer
        self.device = args.device
        #self.TIME = args.TIME

        self.optimizer = self.get_optimizer(self.model)

    def get_optimizer(self, model):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer

    @logger.catch()
    def train_model(self):
        train_dataset_loader = DataLoader(dataset=self.dataset.behavior_dataset(),
                                          batch_size=self.batch_size,
                                          shuffle=True)

        # best_result = np.zeros(len(self.topk) * len(self.metrics))
        best_result = 0
        best_dict = {}
        best_epoch = 0
        best_model = None
        final_test = None
        for epoch in range(int(self.epochs)):
            self.model.train()
            test_metric_dict, validate_metric_dict = self._train_one_epoch(train_dataset_loader, epoch)

    def _train_one_epoch(self, behavior_dataset_loader, epoch):
        #start_time = time.time()
        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=len(behavior_dataset_loader),
                desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
            )
        )
        total_loss = 0.0
        batch_no = 0
        for batch_index, batch_data in behavior_dataset_iter:
            #start = time.time()
            batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(batch_data)
            loss.backward()
            self.optimizer.step()
            batch_no = batch_index + 1
            total_loss += loss.item()
        total_loss = total_loss / batch_no

        #self.train_writer.add_scalar('total Train loss', total_loss, epoch + 1)
        #epoch_time = time.time() - start_time
        #logger.info('epoch %d %.2fs Train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

        #validate
        
        # validate_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.validate_dataset(),
        #                                      self.dataset.validation_interacts, self.dataset.validation_gt_length
        #                                      )
        # test
        test_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.test_dataset(),
                                         self.dataset.test_interacts, self.dataset.test_gt_length
                                         )
        print(test_metric_dict)
        return None, None

    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)
        self.model.eval()
        #start_time = time.time()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.dataset.train_behavior_dict['buy']
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(self.device)
            #start = time.time()
            scores = self.model.full_predict(batch_data)

            for index, user in enumerate(batch_data):
                user_score = scores[index]
                items = train_items.get(str(user.item()), None)
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                gt_items = gt_interacts[str(user.item())]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        
        return metric_dict

    def calculate_result(self, topk_list, gt_len):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = '{}@{}'.format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.model_path, self.model_name + self.TIME + '.pth'))
