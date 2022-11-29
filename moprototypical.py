"""
An implementation of the "Prototypical Networks for Few-shot Learning" in PyTorch, 
trained and tested on bearing fault diagnosis probelm.

Data: 2022/11/13
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
"""
import argparse
import logging
import warnings
import os
import time
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Utils.logger import setlogger
from tqdm import *

from PrepareData.CWRU import CWRUloader
from PrepareData.PU import PUloader
#from PrototypicalNets import CNN1D
from ContrastiveNets import CNN1D, LeNet, AlexNet, ResNet18
from Datasets import PrototypicalData
from Sampler import BatchSampler
from Loss.PrototypicalLoss import prototypical_loss as loss_fn

import Utils.utilis as utils

# ===== Define argments =====
def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of Prototypical Neural Networks')

    # log files
    parser.add_argument("--log_file", type=str, default="./logs/MoPrototypical.log", help="log file path")

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
    parser.add_argument("--backbone", type=str, default="CNN1D", choices=["ResNet1D", "ResNet2D", "MLPNet", "CNN1D"])
    # if   backbone in ("ResNet1D", "CNN1D"),  data shape: (batch size, 1, 1024)
    # elif backbone == "ResNet2D",             data shape: (batch size, 3, 32, 32)
    # elif backbone == "MLPNet",               data shape: (batch size, 1024)
    parser.add_argument("--savemodel", type=bool, default=False, help="whether save pre-trained model in the classification task")
    parser.add_argument("--pretrained", type=bool, default=False, help="whether use pre-trained model in transfer learning tasks")


    parser.add_argument("--n_train", type=int, default=500, help="The number of training data per class")
    parser.add_argument("--n_val", type=int, default=200, help="the number of validation data per class")
    parser.add_argument("--n_test", type=int, default=200, help="the number of test data per class")
    parser.add_argument("--support", type=int, default=5, help="the number of support set per class")
    parser.add_argument("--query", type=int, default=5, help="the number of query set per class")
    parser.add_argument("--episodes", type=int, default=80, help="the number of episodes per epoch")

    parser.add_argument("--split", type=int, default=4, help="split batch normalization, split >= 1 and \in R")
    parser.add_argument("--m", type=float, default=0.99, help="momentum of updating encoder support")
    parser.add_argument("--showstep", type=int, default=50, help="show training history every 'showstep' steps")

    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['step', 'exp', 'stepLR', 'cos', 'fix'], help='the learning rate schedule')
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='60, 120', help='the learning rate decay for step and stepLR')

    args = parser.parse_args()
    return args

# ===== Load Data =====
def loaddata(args):
    if args.source_dataname == "CWRU":
        source_data = CWRUloader(args, args.s_load, args.s_label_set, args.n_train)
    elif args.source_dataname == "PU":
        source_data = PUloader(args, args.s_load, args.s_label_set, args.n_train)
    
    if args.target_dataname == "CWRU":
        target_data = CWRUloader(args, args.t_load, args.t_label_set, args.n_val+args.n_test)
    elif args.target_dataname == "PU":
        target_data = PUloader(args, args.t_load, args.t_label_set, args.n_val+args.n_test)

    val_data = {key:target_data[key][:args.n_val] for key in target_data.keys()}
    test_data = {key:target_data[key][-args.n_test:] for key in target_data.keys()}

    print("Data size of training sample per class: ", source_data[0].shape)
    print("Data size of validation sample per class: ", val_data[0].shape)
    print("Data size of test sampler per class: ", test_data[0].shape)

    # convert the data format from dictionary to tensor
    source_data = PrototypicalData.ProtitypicalData(source_data)
    val_data = PrototypicalData.ProtitypicalData(val_data)
    test_data = PrototypicalData.ProtitypicalData(test_data)

    # source_data.y: all source data labels
    source_sampler = BatchSampler.BatchSampler(source_data.y, args.support, args.query, args.episodes)
    val_sampler = BatchSampler.BatchSampler(val_data.y, args.support, args.query, args.episodes)
    test_sampler = BatchSampler.BatchSampler(test_data.y, args.support, args.query, episodes=args.episodes)

    source_loader = DataLoader(source_data, batch_sampler=source_sampler)
    val_loader = DataLoader(val_data, batch_sampler=val_sampler)
    test_loader = DataLoader(test_data, batch_sampler=test_sampler)
    
    return source_loader, val_loader, test_loader

# ===== Split batch normalization =====
class SplitBatchNorm1D(nn.BatchNorm1d):
    '''
    Adapted from: https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=bNd_Q_Osi0SO
    '''
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
    
    def forward(self, input):
        # (batch_size, 1, signal_length)
        if len(input.size()) == 3:
            N, C, L = input.shape
            if self.training or not self.track_running_stats:
                running_mean_split = self.running_mean.repeat(self.num_splits)
                running_var_split = self.running_var.repeat(self.num_splits)
                outcome = nn.functional.batch_norm(
                    input.view(-1, C * self.num_splits, L), running_mean_split, running_var_split, 
                    self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                    True, self.momentum, self.eps).view(N, C, L)
                self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
                self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
                return outcome
            else:
                return nn.functional.batch_norm(
                    input, self.running_mean, self.running_var, 
                    self.weight, self.bias, False, self.momentum, self.eps)
        elif len(input.size()) == 2:
            N, L = input.shape
            if self.training or not self.track_running_stats:
                running_mean_split = self.running_mean.repeat(self.num_splits)
                running_var_split = self.running_var.repeat(self.num_splits)
                outcome = nn.functional.batch_norm(
                    input.view(-1, self.num_splits, L), running_mean_split, running_var_split, 
                    self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                    True, self.momentum, self.eps).view(N, L)
                self.running_mean.data.copy_(running_mean_split.view(self.num_splits).mean(dim=0))
                self.running_var.data.copy_(running_var_split.view(self.num_splits).mean(dim=0))
                return outcome
            else:
                return nn.functional.batch_norm(
                    input, self.running_mean, self.running_var, 
                    self.weight, self.bias, False, self.momentum, self.eps)
        else:
            raise Exception("{} dimentinal signal time masking is not implemented, please try 1 or 2 dimentional signal.")

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

class MoPrototypical(nn.Module):
    def __init__(self, m = 0.99, bn_splits = 4):
        super(MoPrototypical, self).__init__()
        self.m = m

        self.norm_layer = partial(SplitBatchNorm1D, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm1d
        # support encoder_s and query encoder_q
        # self.encoder_s = CNN1D.CNN1D(dim=64, norm_layer=self.norm_layer)
        # self.encoder_q = CNN1D.CNN1D(dim=64, norm_layer=self.norm_layer)
        self.encoder_s = ResNet18.resnet18()
        self.encoder_q = ResNet18.resnet18()
        self.norm = Normalize(2)

        for param_s, param_q in zip(self.encoder_s.parameters(), self.encoder_q.parameters()):
            param_s.data.copy_(param_q.data)
            param_s.requires_grad = False
        
    @torch.no_grad()
    def _momentum_update_query_encoder(self):
        '''
        Momentum update of the query encoder
        '''
        for param_s, param_q in zip(self.encoder_s.parameters(), self.encoder_q.parameters()):
            param_s.data = param_s.data * self.m + param_q.data * (1. - self.m)

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def split_support_query(self, y, n_query):
        classes = torch.unique(y)

        # ===================================
        # to fastly test the performance, the number of support samples are equal
        # ===================================
        n_support = n_query

        support_indices = list(map(lambda c: y.eq(c).nonzero()[:n_support].squeeze(1), classes))
        query_indices = list(map(lambda c: y.eq(c).nonzero()[n_query:], classes))
        return support_indices, torch.stack(query_indices).view(-1)

    def forward(self, x, y):
        with torch.no_grad():
            self._momentum_update_query_encoder()
        
        support_out = self.encoder_s(x)
        query_out = self.encoder_q(x)

        # L2 normaliza
        support_out = self.norm(support_out)
        query_out = self.norm(query_out)

        n_classes = len(torch.unique(y))
        n_query = 5
        support_indices, query_indices = self.split_support_query(y, n_query)

        prototypes = torch.stack([support_out[idx_list].mean(0) for idx_list in support_indices])

        # fetch query encoder output
        query_samples = query_out[query_indices]

        # Compute distances
        dists = self.euclidean_dist(query_samples, prototypes) # [n_samples, n_classes], i.e. n_samples = n_classes * n_query

        log_prob = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1) # [10, 5, 10]

        target_inds = torch.arange(0, n_classes, device='cuda:0')
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, 5, 1).long()

        loss_query = -log_prob.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_prob.max(2)
        acc_query = y_hat.eq(target_inds.squeeze(2)).float().mean() # query set accuracy

        return loss_query,  acc_query

# ===== Evaluate the model =====
def Test(Net, dataloader):
    Net.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_iter = iter(dataloader)
    acc_his, loss_his = [], []
    for x,y in data_iter:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            loss, acc = Net(x, y)
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
    Net = MoPrototypical(m=args.m,bn_splits=4)

    # Define optimizer and learning rate decay
    parameter_list = [{"params": Net.parameters(), "lr": args.lr}]
    optimizer, lr_scheduler = utils.optimizer(args, parameter_list)

    Net.to(device)

    # train
    best_acc = 0.0
    meters = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

    for epoch in range(args.max_epoch):
        Net.train()
        train_iter = iter(source_loader) # len(train_iter) = episodes
        acc_his, loss_his = [], []
        for x,y in tqdm(train_iter, ncols = 70, leave=False):
            # move to GPU if available
            x = x.to(device)  # [class*(num_support + num_querry), sample_length]
            y = y.to(device)  # [class*(num_support + num_querry)]


            loss, acc = Net(x, y)

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
        val_acc, val_loss = Test(Net, val_loader)
        meters["val_acc"].append(val_acc)
        meters["val_loss"].append(val_loss)

        # recording the best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(Net.state_dict(), "./checkpoints/moprototypical_best.tar")

        # print training history
        if epoch % args.showstep == 0:
            logging.info("Epoch: {:>3}/{}, current lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, train_acc: {:6.2f}%, val_acc: {:6.2f}%".format(
                epoch+1, args.max_epoch, optimizer.param_groups[0]['lr'], train_loss, val_loss, train_acc*100, val_acc*100))

    utils.save_log(meters, "./History/MoPrototypical.pkl")

    # ===================
    #        Test
    # ===================
    # load parameters
    params = torch.load("./checkpoints/moprototypical_best.tar")
    Net.load_state_dict(params)
    test_acc, _ = Test(Net, test_loader)
    logging.info("===> Best validation accuracy: {:6.2f}%".format(best_acc*100))
    logging.info("===> Test results with best validation model: {:6.2f}%".format(test_acc*100))
    logging.info("="*15+"Done!"+"="*15)

    # # if run many rounds, using the following commands to save 
    # # the test accuracy with best validation model
    # if os.path.exists("./History/5_round_mopro.txt"):
    #     results = np.loadtxt("./History/5_round_mopro.txt")
    #     results = np.append(results, test_acc)
    #     np.savetxt("./History/5_round_mopro.txt", results)
    # else:
    #     result = np.array([test_acc])
    #     np.savetxt("./History/5_round_mopro.txt", result)

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