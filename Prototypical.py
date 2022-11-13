"""
An implementation of the "Prototypical Networks for Few-shot Learning" in PyTorch, 
trained and tested on bearing fault diagnosis probelm.

Data: 2022/11/
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
from Datasets import PrototypicalData
from Sampler import BatchSampler
from Loss.PrototypicalLoss import prototypical_loss as loss_fn

import Utils.utilis as utils

# ===== Define argments =====
def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of Domain Adversarial Neural Networks')

    # log files
    parser.add_argument("--log_file", type=str, default="./logs/Siamese.log", help="log file path")

    # dataset information
    parser.add_argument("--datadir", type=str, default="/home/xiaohan/codelab/datasets", help="data directory")
    parser.add_argument("--source_dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--target_dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--s_load", type=int, default=3, help="source domain working condition")
    parser.add_argument("--t_load", type=int, default=2, help="target domain working condition")
    parser.add_argument("--s_label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="source domain label set")
    parser.add_argument("--t_label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="target domain label set")

    # pre-processing
    parser.add_argument("--fft", type=bool, default=False, help="FFT preprocessing")
    parser.add_argument("--window", type=int, default=128, help="time window, if not augment data, window=1024")
    parser.add_argument("--normalization", type=str, default="0-1", choices=["None", "0-1", "mean-std"], help="normalization option")


    # backbone
    parser.add_argument("--backbone", type=str, default="ResNet1D", choices=["ResNet1D", "ResNet2D", "MLPNet", "CNN1D"])
    # if   backbone in ("ResNet1D", "CNN1D"),  data shape: (batch size, 1, 1024)
    # elif backbone == "ResNet2D",             data shape: (batch size, 3, 32, 32)
    # elif backbone == "MLPNet",               data shape: (batch size, 1024)
    parser.add_argument("--savemodel", type=bool, default=False, help="whether save pre-trained model in the classification task")
    parser.add_argument("--pretrained", type=bool, default=False, help="whether use pre-trained model in transfer learning tasks")


    parser.add_argument("--n_train", type=int, default=200, help="The number of training data per class")
    parser.add_argument("--n_val", type=int, default=30, help="the number of validation data per class")
    parser.add_argument("--n_test", type=int, default=50, help="the number of test data per class")
    parser.add_argument("--support", type=int, default=5, help="the number of support set per class")
    parser.add_argument("--query", type=int, default=15, help="the number of query set per class")
    parser.add_argument("--episodes", type=int, default=80, help="the number of episodes per epoch")

    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='stepLR', choices=['step', 'exp', 'stepLR', 'fix'], help='the learning rate schedule')
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='30, 120', help='the learning rate decay for step and stepLR')

    args = parser.parse_args()
    return args

# ===== Load Data =====
def loaddata(args):
    if args.source_dataname == "CWRU":
        source_data = CWRUloader(args, args.s_load, args.s_label_set, args.n_train)
    
    if args.target_dataname == "CWRU":
        target_data = CWRUloader(args, args.t_load, args.t_label_set, args.n_val+args.n_test)
    val_data = {key:target_data[key][:args.n_val] for key in target_data.keys()}
    test_data = {key:target_data[key][-args.n_test:] for key in target_data.keys()}

    print("Data size of training sample per class: ", source_data[0].shape)
    print("Data size of validation sample per class: ", val_data[0].shape)
    print("Data size of test sampler per class: ", test_data[0].shape)

    source_data = PrototypicalData.ProtitypicalData(source_data)
    val_data = PrototypicalData.ProtitypicalData(val_data)
    test_data = PrototypicalData.ProtitypicalData(test_data)

    source_sampler = BatchSampler.BatchSampler(source_data.y, args.support, args.query, args.episodes)
    val_sampler = BatchSampler.BatchSampler(val_data.y, args.support, args.query, args.episodes)
    test_sampler = BatchSampler.BatchSampler(test_data.y, args.support, args.query, episodes=args.episodes)

    source_loader = DataLoader(source_data, batch_sampler=source_sampler)
    val_loader = DataLoader(val_data, batch_sampler=val_sampler)
    test_loader = DataLoader(test_data, batch_sampler=test_sampler)

    print("========== Loading dataset down! ==========")
    
    return source_loader, val_loader, test_loader

# ===== Evaluate the model =====
def Test(args, Net, dataloader):
    Net.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    correct_num, error_num = 0, 0

    test_iter = iter(dataloader)
    acc_his, loss_his = [], []
    for x,y in test_iter:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            outputs = Net(x)
            loss, acc = loss_fn(outputs, target=y, n_support=args.support)
            acc_his.append(acc.item())
            loss_his.append(loss.item())
    avg_acc = np.mean(acc_his)
    avg_loss = np.mean(loss_his)

    return avg_acc, avg_loss

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

        # validation
        val_acc, val_loss = Test(args, Net, val_loader)
        meters["val_acc"].append(val_acc)
        meters["val_loss"].append(val_loss)

        # recording the best accuracy
        if val_acc > best_acc:
            best_acc = acc

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