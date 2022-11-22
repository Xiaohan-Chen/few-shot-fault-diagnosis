'''
A linear classifier protocol for self-supervised learning pre-trained model.

Data: 2022/11/22
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
from Datasets import ContrastiveData

import Utils.utilis as utils

# ===== Define argments =====
def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of MoCo v2.')

    # log files
    parser.add_argument("--log_file", type=str, default="./logs/LinearProtocol.log", help="log file path")

    # dataset information
    parser.add_argument("--datadir", type=str, default="/home/xiaohan/codelab/datasets", help="data directory")
    parser.add_argument("--dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--load", type=int, default=1, help="source domain working condition")
    parser.add_argument("--label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="source domain label set")

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

    # trainig data
    parser.add_argument("--n_train", type=int, default=50, help="The number of training data per class")
    parser.add_argument("--n_test", type=int, default=50, help="the number of test data per class")

    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'exp', 'stepLR', 'fix'], help='the learning rate schedule')
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default=[60, 120], help='the learning rate decay for step and stepLR')

    args = parser.parse_args()
    return args

# ===== Load Data =====
def loaddata(args):
    if args.dataname == "CWRU":
        data = CWRUloader(args, args.load, args.label_set, args.n_train + args.n_test)

    print("Data size of training sample per class: ", data[0].shape)

    data_train = {key:data[key][args.n_train:] for key in data.keys()}
    data_test = {key:data[key][-args.n_test:] for key in data.keys()}

    train_data = ContrastiveData.ContrastiveData(data_train)
    test_data = ContrastiveData.ContrastiveData(data_test)

    # dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    print("========== Loading dataset down! ==========")
    
    return train_loader, test_loader

# ===== Load pre-trained model via self-supervised learning =====
def Load_model(args):
    Model = CNN1D.CNN1D()

    if os.path.isfile(args.checkpoint):
        # read parameters
        params = torch.load(args.checkpoint, map_location="cpu")
        Model.load_state_dict(params)

        # freeze all layers but the last fc
        for name, param in Model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # init the fc layer
        Model.fc.weight.data.normal_(mean=0.0, std=0.01)
        Model.fc.bias.data.zero_()
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
def Train(args):
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
    train_loader, test_loader = loaddata(args)

    # load the pre-trained model
    model = Load_model(args).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # optimize only the fc layer
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters == 2) # fc.weight, fc.bias
    optimizer, lr_scheduler = utils.optimizer(args, parameters)

    # train
    best_acc = 0.0
    meters = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(args.max_epoch):
        # BatchNorm in train mode may revise running mean/std
        # therefore, transform the model to eval mode
        model.eval()
        for x,y in tqdm(train_loader, ncols = 70, leave=False):
            # move to GPU if available
            x, y = x.to(device), y.to(device)
            
            # compute output
            output = model(x)
            loss = criterion(output, y)

            # clear previous gradients, compute gradients
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()
        train_acc = utils.accuracy(output, y)
        meters["loss"].append(loss.item())
        meters["train_acc"].append(train_acc)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # test
        test_acc = Test(model, test_loader)
        meters["test_acc"].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
        
        # print training history
        logging.info("Epoch: {:>3}/{}, train_loss: {:.4f}, train_acc: {:6.2f}%, test_acc: {:6.2f}%".format(
            epoch+1, args.max_epoch, loss.item(), train_acc*100, test_acc*100))

    utils.save_log(meters, "./History/LinearProtocol.pkl")

    logging.info("Best accuracy: {:.4f}".format(best_acc))
    logging.info("="*15+"Done!"+"="*15)




if __name__ == "__main__":

    if not os.path.exists("./History"):
        os.makedirs("./History")

    args = parse_args()

    # set the logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger(args.log_file)

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    
    Train(args)