"""
An implementation of the "Siamese Neural Networks for One-shot Image Recognition" in PyTorch, 
trained and tested on bearing fault diagnosis probelm.

Data: 2022/11/08
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
from SiameseNets import CNN1D
from Datasets import SiameseData
import Utils.utilis as utils

# ===== Define argments =====
def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of Domain Adversarial Neural Networks')

    # log files
    parser.add_argument("--log_file", type=str, default="./logs/Siamese.log", help="log file path")

    # dataset information
    parser.add_argument("--datadir", type=str, default="/home/xiaohan/codelab/datasets", help="data directory")
    parser.add_argument("--source_dataname", type=str, default="CWRU", choices=["CWRU"], help="choice a dataset")
    parser.add_argument("--target_dataname", type=str, default="CWRU", choices=["CWRU"], help="choice a dataset")
    parser.add_argument("--s_load", type=int, default=3, help="source domain working condition")
    parser.add_argument("--t_load", type=int, default=2, help="target domain working condition")
    parser.add_argument("--s_label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="source domain label set")
    parser.add_argument("--t_label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="target domain label set")

    # pre-processing
    parser.add_argument("--fft", type=bool, default=False, help="FFT preprocessing")
    parser.add_argument("--window", type=int, default=128, help="time window, if not augment data, window=1024")
    parser.add_argument("--normalization", type=str, default="0-1", choices=["None", "0-1", "mean-std"], help="normalization option")


    # backbone
    parser.add_argument("--backbone", type=str, default="CNN1D", choices=["ResNet1D",  "MLPNet", "CNN1D"])
    # if   backbone in ("ResNet1D", "CNN1D"),  data shape: (batch size, 1, 1024)
    # elif backbone == "ResNet2D",             data shape: (batch size, 3, 32, 32)
    # elif backbone == "MLPNet",               data shape: (batch size, 1024)
    parser.add_argument("--savemodel", type=bool, default=False, help="whether save pre-trained model in the classification task")
    parser.add_argument("--pretrained", type=bool, default=False, help="whether use pre-trained model in transfer learning tasks")

    # training set
    parser.add_argument("--support", type=int, default=200, help="the number of training samples per class")
    # test set
    parser.add_argument("--querry", type=int, default=100, help="the number of test samples per class")
    parser.add_argument("--shots", type=int, default=1, help="the number of test samples per class in querry set")

    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='stepLR', choices=['step', 'exp', 'stepLR', 'fix'], help='the learning rate schedule')
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='30, 120', help='the learning rate decay for step and stepLR')

    args = parser.parse_args()
    return args

# ===== Load Data =====
def loaddata(args):
    if args.source_dataname == "CWRU":
        source_data = CWRUloader(args, args.s_load, args.s_label_set, args.support)
    
    if args.target_dataname == "CWRU":
        target_data = CWRUloader(args, args.t_load, args.t_label_set, args.querry)

    print("Data size of training sample per class: ", source_data[0].shape)
    print("Data size of test sample per class: ", target_data[0].shape)

    source_data = SiameseData.SiameseTrain(source_data)
    target_data = SiameseData.SiameseTest(target_data)

    train_loader = DataLoader(source_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(target_data, batch_size=len(args.t_label_set), shuffle=False)

    print("========== Loading dataset down! ==========")
    
    return train_loader, test_loader

# ===== Evaluate the model =====
def Test(Net, dataloader):
    Net.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    correct_num, error_num = 0, 0

    for i, (x1, x2) in enumerate(dataloader):
        # move to GPU if available
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        with torch.no_grad():
            outputs = Net(x1, x2).data.cpu().numpy()
            pre = np.argmax(outputs)
            if pre == 0:
                correct_num += 1
            else:
                error_num += 1
        
    return correct_num, error_num

# ===== Train the model =====
def Train(args):
    # Consider the gpu or cpu condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(device_count))
        assert args.batch_size % device_count == 0, "batch size should be divided by device count"
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1
        logging.info('using {} cpu'.format(device_count))
    
    # load datasets
    train_loader, test_loader = loaddata(args)
    
    # load the Siamese Network
    SiameseNet = CNN1D.CNN1D()

    # Define optimizer and learning rate decay
    parameter_list = [{"params": SiameseNet.parameters(), "lr": args.lr}]
    optimizer, lr_scheduler = utils.optimizer(args, parameter_list)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    SiameseNet.to(device)

    # train
    best_acc = 0.0
    meters = {"acc": [], "loss": []}

    for epoch in range(args.max_epoch):
        SiameseNet.train()
        with tqdm(total=len(train_loader), leave=False) as pbar:
            for i, (x1, x2, y) in enumerate(train_loader):
                # move to GPU if available
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                pre = SiameseNet(x1, x2)
                loss = loss_fn(pre, y.unsqueeze(1))

                # clear previous gradients, compute gradients
                optimizer.zero_grad()
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

        pbar.update()

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # evaluate
        correct_num, error_num = Test(SiameseNet, test_loader)
        acc = correct_num / (correct_num + error_num)

        if acc > best_acc:
            best_acc = acc
        
        # print training history
        logging.info("Epoch: {:>3}/{}, loss: {:.4f}, accuracy: {:6.2f}%".format(epoch+1, args.max_epoch, loss.item(), acc*100))

        # recording history data
        meters["acc"].append(acc)
        meters["loss"].append(loss.item())

    utils.save_log(meters, "./History/Siamese.pkl")

    logging.info("Best accuracy: {:.4f}".format(best_acc))
    logging.info("="*15+"Done!"+"="*15)

if __name__ == "__main__":

    if not os.path.exists("./History"):
        os.makedirs("./History")

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    args = parse_args()

    # set the logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger(args.log_file)

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    
    Train(args)