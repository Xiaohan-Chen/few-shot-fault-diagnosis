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
    parser.add_argument("--log_file", type=str, default="./logs/MoCo.log", help="log file path")

    # dataset information
    parser.add_argument("--datadir", type=str, default="/home/xiaohan/codelab/datasets", help="data directory")
    parser.add_argument("--dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--load", type=int, default=3, help="source domain working condition")
    parser.add_argument("--label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="source domain label set")
    parser.add_argument("--test_dataname", type=str, default="CWRU", help="test dataset name")
    parser.add_argument("--test_load", type=int, default=2, help="test working condition")
    parser.add_argument("--test_label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="test label set")

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
    parser.add_argument("--n_test", type=int, default=50, help="the number of test data per class")

    # moco
    parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco_k', default=4096, type=int, help='queue size, number of negative keys')
    parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco_t', default=0.1, type=float, help='softmax temperature')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

    # knn monitor
    parser.add_argument('--knn-k', default=50, type=int, help="k in knn mornitor")
    parser.add_argument('--knn-t', default=0.1, type=float, help="softmax temperature in knn monitor, could be different with moco-t")

    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'exp', 'stepLR', 'fix'], help='the learning rate schedule')
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default=[60, 120], help='the learning rate decay for step and stepLR')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')

    args = parser.parse_args()
    return args

# ===== Load Data =====
def loaddata(args):
    if args.dataname == "CWRU":
        data = CWRUloader(args, args.load, args.label_set, args.n_train)
    
    if args.test_dataname == "CWRU":
        test_data = CWRUloader(args, args.test_load, args.test_label_set, args.n_test)    

    print("Data size of training sample per class: ", data[0].shape)
    print("Data size of test sample per class: ", test_data[0].shape)

    train_data = ContrastiveData.ContrastivePair(data)
    memory_data = ContrastiveData.ContrastiveData(data)
    test_data = ContrastiveData.ContrastiveData(test_data)

    # dataloader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    print("========== Loading dataset down! ==========")
    
    return train_loader, memory_loader, test_loader

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

# ===== Define encoder =====
class ModelBase(nn.Module):
    '''
    Encoder
    '''
    def __init__(self, feature_dim=128, bn_splits=8) -> None:
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm1D, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        self.net = CNN1D.CNN1D(dim=feature_dim, norm_layer=norm_layer)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

# ===== Define MoCo model =====
class ModelMoCo(nn.Module):
    '''
    Reference: https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    '''
    def __init__(self, dim=128, k=4096, m=0.99, T=0.1, bn_splits=8, symmetric=False):
        super(ModelMoCo, self).__init__()

        self.k = k
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, bn_splits=bn_splits)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) # initialize
            param_k.requires_grad = False # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(dim, k)) # [128, 4096]
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''
        Momentum update of the key encoder
        '''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.k  # move pointer

        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, x_q, x_k):
        # compute query features
        q = self.encoder_q(x_q)  # queries: NxC [batch size, dim]
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            x_k_, idx_unshuffle = self._batch_shuffle_single_gpu(x_k)

            k = self.encoder_k(x_k_)  # keys: NxC [batch size, dim]
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # [batch size, 1]
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) # [batch size, dim]

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) # [batch size, dim+1]

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, x1, x2):
        """
        Input:
            x_q: a batch of query samples
            x_k: a batch of key samples
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(x1, x2)
            loss_21, q2, k1 = self.contrastive_loss(x2, x1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(x1, x2)

        self._dequeue_and_enqueue(k)

        return loss

# ===== Lr scheduler for training =====
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.steps:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank) # [batch size, dim] x [dim, sample size] = 
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)  # return a descending order indices
    return pred_labels

# ===== Test =====
def Test(args, encoder, memory_loader, test_loader):
    encoder.eval()
    classes = memory_loader.dataset.classes
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    feature_bank = []
    correct_counter, total_counter = 0, 0
    with torch.no_grad():
        for data, _ in tqdm(memory_loader, ncols = 70, leave=False):
            data = data.to(device)
            feature  = encoder(data)
            feature = F.normalize(feature, dim=1) # batch normalization
            feature_bank.append(feature)
        
        # [dim, sample size]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [sample size,]
        feature_labels = torch.tensor(memory_loader.dataset.y, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        for data, target in tqdm(test_loader, ncols = 70, leave=False):
            data, target = data.to(device), target.to(device)
            feature = encoder(data)
            feature = F.normalize(feature, dim=1) # batch normalization

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)
            total_counter += data.size(0)
            correct_counter += (pred_labels[:,0] == target).float().sum().item()
    return correct_counter / total_counter * 100
        

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
    train_loader, memory_loader, test_loader = loaddata(args)
    
    # load Model and move to GPU
    Net = ModelMoCo(dim=args.moco_dim,
                    k=args.moco_k,
                    m=args.moco_m,
                    T=args.moco_t,
                    bn_splits=8,
                    symmetric=False).to(device)

    # define optimizer
    optimizer = torch.optim.SGD(Net.parameters(),
                                lr=args.lr,
                                weight_decay=args.wd,
                                momentum=0.9)

    # train
    # k+1-way contrastive accuracy and loss
    meters = {"train_loss": [], "test_acc": []}

    for epoch in range(args.max_epoch):
        Net.train()
        # define lr scheduler
        adjust_learning_rate(optimizer, epoch, args)
        loss_his = []
        for x1,x2 in tqdm(train_loader, ncols = 70, leave=False):
            # move to GPU if available
            x1, x2 = x1.to(device), x2.to(device)

            loss = Net(x1, x2)

            # clear previous gradients, compute gradients
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # accuracy and loss of per episode
            loss_his.append(loss.item())
            test_acc = Test(args, Net.encoder_q, memory_loader, test_loader)

        train_loss = np.mean(loss_his)
        meters["train_loss"].append(train_loss)
        meters["test_acc"].append(test_acc)

        # print training history
        logging.info("Epoch: {:>3}/{}, current lr: {:.6f}, train_loss: {:.4f}, test_acc: {:6.2f}%".format(
            epoch+1, args.max_epoch, optimizer.param_groups[0]['lr'], train_loss, test_acc))

    utils.save_log(meters, "./History/MoCo.pkl")

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