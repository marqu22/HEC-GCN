#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author:
# @Date  : 2021/11/1 15:25
# @Desc  :
import argparse
import os
import random
import time

import numpy as np
import torch
from loguru import logger
import ast
from data_set import DataSet
from model import HEC_GCN

from trainer import Trainer
import json




if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--node_dropout', type=float, default=0.75)
    parser.add_argument('--message_dropout', type=float, default=0.25)
    parser.add_argument('--dim_qk', type=int, default=32)
    parser.add_argument('--dim_v', type=int, default=64)
    parser.add_argument('--omega', type=float, default=1)

    parser.add_argument('--data_name', type=str, default='tmall', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')

    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--topk', type=list, default=[10, 30, 50, 100, 200], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.0005, help='')
    parser.add_argument('--decay', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=100, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='a_tmall_base.pth', help='')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--pt_loop', type=int, default=50, help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--seed', type=int, default=2021, help='')
    
    ##added params
    parser.add_argument('--layers_nums', type=str, default="[2, 2, 2]", help='Each layer can have a value between 1 and 3')
    parser.add_argument('--cl_coefficient', type=str, default="[0.5, 2, 0.5]", help='The sum is 3, and each can have a value from 0 to 2.5 in increments of 0.5.')  #lambda
    parser.add_argument('--loss_coefficient', type=str, default="[1, 1, 1]", help='equal') #in paper 
    parser.add_argument('--hyper_dropout', type=float, default=0.1, help='') #not used parameter in codes.
    parser.add_argument('--hyper_nums', type=float, default=32, help='') #in the paper 32 is best
    parser.add_argument('--tau', type=float, default=0.1, help='') #in the paper 0.1 is best
    parser.add_argument('--alpha', type=float, default=0.3, help='0.1 ~ 0.5') #
    args = parser.parse_args()
    args.cl_coefficient = ast.literal_eval(args.cl_coefficient)
    args.loss_coefficient = ast.literal_eval(args.loss_coefficient)
    args.layers_nums = ast.literal_eval(args.layers_nums)

    
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # True can improve train speed
        torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    tp = args.tp
    if args.data_name == 'tmall':
        args.data_path = f'./data/tmall/'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
    elif args.data_name == 'beibei':
        args.data_path = f'./data/beibei/'
        args.behaviors = ['view', 'cart', 'buy']
    elif args.data_name == 'jdata':
        args.data_path = f'./data/jdata/'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
    elif args.data_name == 'tenrec':
        args.data_path = f'./data/tenrec/'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
    elif args.data_name == 'taobao':
        args.data_path = f'./data/taobao/'
        args.behaviors = ['view', 'cart', 'buy']
    else:
        raise Exception('data_name cannot be None')

    start = time.time()
    dataset = DataSet(args)
    model = HEC_GCN(args, dataset)

    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    result = trainer.evaluate(0, 1, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length)
    direct = f'./results/{args.tp}/'
    os.makedirs(direct, exist_ok=True)

    if os.path.isfile(direct + args.data_name + '.csv') == False:
        with open(direct + args.data_name + '.csv', 'a') as f:
            temp_line = ["config", "seed"]
            for i in result:
                temp_line.append(str(i))
            f.write(','.join(temp_line) + '\n')
    
    with open(direct + args.data_name + '.csv', 'r') as f:
        nconf = len(f.readlines())
    
    os.makedirs(f"./conf/{args.data_name}", exist_ok=True)
    json.dump(vars(args), open(f"./conf/{args.data_name}/{nconf}.json", 'w'))
    
    with open(direct + args.data_name + '.csv', 'a') as f:
        temp_line = [str(nconf),str(args.seed)]
        for i in result:
            temp_line.append(str(result[i].item()))
        f.write(','.join(temp_line) + '\n')