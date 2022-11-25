'''
Self-supervised learning to pre-train the backbone, freeze the backbone and
test the performance on few-shot problems via prototypical method.

Data: 2022/11/24
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
'''

import argparse
import logging
import warnings
import os
import math
import time
import numpy as np
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Utils.logger import setlogger
from tqdm import *

from PrepareData.CWRU import CWRUloader
from ContrastiveNets import CNN1D
from Datasets import PrototypicalData
from Sampler import BatchSampler
from Loss.PrototypicalLoss import prototypical_loss as loss_fn

import Utils.utilis as utils

# ===== Define argments =====
def parse_args():
    parser = argparse.ArgumentParser(description='Self-supervised learning for few-shot learning test.')

    # dataset information
    parser.add_argument("--datadir", type=str, default="/home/xiaohan/codelab/datasets", help="data directory")
    parser.add_argument("--dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--load", type=int, default=2, help="source domain working condition")
    parser.add_argument("--label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="domain label set")

    # pre-processing
    parser.add_argument("--fft", type=bool, default=False, help="FFT preprocessing")
    parser.add_argument("--window", type=int, default=128, help="time window, if not augment data, window=1024")
    parser.add_argument("--normalization", type=str, default="None", choices=["None", "0-1", "mean-std"], help="normalization option")
    # normalizations are conduceted in data augmentation


    # backbone
    parser.add_argument("--backbone", type=str, default="CNN1D", choices=["ResNet1D", "ResNet2D", "MLPNet", "CNN1D"])
    # if   backbone in ("ResNet1D", "CNN1D"),  data shape: (batch size, 1, 1024)
    # elif backbone == "ResNet2D",             data shape: (batch size, 3, 32, 32)
    # elif backbone == "MLPNet",               data shape: (batch size, 1024)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/CNN1D.tar", help="pre-trained model parameters")

    # dataset
    parser.add_argument("--n_sample", type=int, default=200, help="test smples per class")
    parser.add_argument("--n_support", type=int, default=5, help="The number of training data per class")
    parser.add_argument("--n_query", type=int, default=5, help="the number of test data per class")
    parser.add_argument("--episodes", type=int, default=60, help="the number of episodes per epoch")

    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'exp', 'stepLR', 'fix'], help='the learning rate schedule')
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='60, 120', help='the learning rate decay for step and stepLR')

    args = parser.parse_args()
    return args

# ===== Load Data =====
def loaddata(args):
    if args.dataname == "CWRU":
        data = CWRUloader(args, args.load, args.label_set, args.n_sample)

    print("Data size of training sample per class: ", data[0].shape)

    dataset = PrototypicalData.ProtitypicalData(data)

    sampler = BatchSampler.BatchSampler(dataset.y, args.n_support, args.n_query, args.episodes)


    # dataloader
    data_loader = DataLoader(dataset, batch_sampler=sampler)

    print("========== Loading dataset down! ==========")
    
    return data_loader

# ===== Load pre-trained model via self-supervised learning =====
def Load_model(args):
    Model = CNN1D.CNN1D()

    if os.path.isfile(args.checkpoint):
        # read parameters
        params = torch.load(args.checkpoint, map_location="cpu")
        Model.load_state_dict(params)

        # freeze all layers
        for name, param in Model.named_parameters():
            param.requires_grad = False

    else:
        raise Exception("Checkpoint {} is not found".format(args.checkpoint))

    print("==> load pre-trained model: {}".format(args.checkpoint))

    return Model

# ===== Evaluate the model =====
def Test(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    correct_num, total_num = 0, 0
    with torch.no_grad():
        for x,y in tqdm(dataloader, ncols = 70, leave=False):
            x, y = x.to(device), y.to(device)
            output = model(x)

            pre = torch.max(output.cpu(), 1)[1].numpy()
            label = y.data.cpu().numpy()
            correct_num += (pre == label).sum()
            total_num += y.size(0)

    return correct_num / total_num

# ===== Train the model =====
def Evaluate(args):
    # Consider the gpu or cpu condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(device_count))
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1
        logging.info('using {} cpu'.format(device_count))
    
    # load datasets
    dataloader = loaddata(args)

    # load the pre-trained model
    model = Load_model(args).to(device)

    model.eval()
    acc_his, loss_his = [], []
    with torch.no_grad():
        for x,y in tqdm(iter(dataloader), ncols = 70, leave=False):
            # move to GPU if available
            x, y = x.to(device), y.to(device)
            
            # compute output
            output = model(x)

            loss, acc = loss_fn(output, target=y, n_support=args.n_support)

            acc_his.append(acc.item())
            loss_his.append(loss.item())
    acc_mean = np.array(acc_his).mean()
    loss_mean = np.array(loss_his).mean()

    acc_var = np.array(acc_his).var()
    loss_var = np.array(loss_his).var()

    print("Test mean accuracy: {}, var: {}.".format(acc_mean, acc_var))
    print("Test mean loss: {}, var: {}".format(loss_mean, loss_var))

if __name__ == "__main__":
    args = parse_args()
    Evaluate(args)