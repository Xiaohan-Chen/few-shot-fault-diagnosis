"""
An implementation of the "Momentum Contrast for Unsupervised Visual Representation Learning" in PyTorch, 
trained and tested on bearing fault diagnosis probelm.

Data: 2022/11/18
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
"""

import argparse
import logging
import warnings
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from Utils.logger import setlogger
from tqdm import *

from PrepareData.CWRU import CWRUloader
from PrototypicalNets import CNN1D
from Datasets import ContrastiveData
from Sampler import BatchSampler
from Loss.PrototypicalLoss import prototypical_loss as loss_fn

import Utils.utilis as utils

# ===== Define argments =====
def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of MoCo v2.')

    # log files
    parser.add_argument("--log_file", type=str, default="./logs/MoCo.log", help="log file path")

    # dataset information
    parser.add_argument("--datadir", type=str, default="/home/xiaohan/codelab/datasets", help="data directory")
    parser.add_argument("--dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--load", type=int, default=3, help="source domain working condition")
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
    parser.add_argument("--savemodel", type=bool, default=False, help="whether save pre-trained model in the classification task")
    parser.add_argument("--pretrained", type=bool, default=False, help="whether use pre-trained model in transfer learning tasks")

    # trainig data
    parser.add_argument("--n_train", type=int, default=200, help="The number of training data per class")
    parser.add_argument("--n_val", type=int, default=30, help="the number of validation data per class")
    parser.add_argument("--n_test", type=int, default=50, help="the number of test data per class")

    # moco
    parser.add_argument('--moco-dim', default=64, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=4096, type=int, help='queue size, number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='stepLR', choices=['step', 'exp', 'stepLR', 'fix'], help='the learning rate schedule')
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='30, 120', help='the learning rate decay for step and stepLR')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')

    args = parser.parse_args()
    return args

# ===== Load Data =====
def loaddata(args):
    if args.source_dataname == "CWRU":
        data = CWRUloader(args, args.load, args.label_set, args.n_train)
    

    print("Data size of training sample per class: ", data[0].shape)

    train_data = ContrastiveData.ContrastivePair(data)
    memory_data = ContrastiveData.ContrastiveData(data)
    test_data = ContrastiveData.ContrastiveData(data)

    # dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    print("========== Loading dataset down! ==========")
    
    return train_data, memory_loader, test_loader

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
    source_loader, val_loader, test_loader = loaddata(args)
    
    # load the Prototypical Network
    Net = CNN1D.CNN1D()

    # Define optimizer and learning rate decay
    parameter_list = [{"params": Net.parameters(), "lr": args.lr}]
    optimizer, lr_scheduler = utils.optimizer(args, parameter_list)

    Net.to(device)

    # train
    best_acc = 0.0
    meters = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

    for epoch in range(args.max_epoch):
        Net.train()
        train_iter = iter(source_loader)
        acc_his, loss_his = [], []
        for x,y in tqdm(train_iter, leave=False):
            # move to GPU if available
            x = x.to(device)
            y = y.to(device)

            outputs = Net(x)

            loss, acc = loss_fn(outputs, target=y, n_support=args.support)

            # clear previous gradients, compute gradients
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # accuracy and loss of per episode
            acc_his.append(acc.item())
            loss_his.append(loss.item())

        train_acc = np.mean(acc_his)
        train_loss = np.mean(loss_his)
        meters["train_acc"].append(train_acc)
        meters["train_loss"].append(train_loss)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()


        # print training history
        logging.info("Epoch: {:>3}/{}, train_loss: {:.4f}, val_loss: {:.4f}, train_acc: {:6.2f}%, val_acc: {:6.2f}%".format(
            epoch+1, args.max_epoch, train_loss, val_loss, train_acc*100, val_acc*100))

    utils.save_log(meters, "./History/Prototypical.pkl")

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