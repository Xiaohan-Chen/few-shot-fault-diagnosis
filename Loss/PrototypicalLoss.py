# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
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


def prototypical_loss(prediction, target, n_support):
    '''
    Reference: https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Args:
    - prediction: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    prediction_cpu = prediction.to('cpu')  # (600, 64)
    target_cpu = target.to('cpu')  # (600)


    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support # number of samples per class - n_support

    def find_support_idxs(c):
        """
        Input a class 'c', return the indexes of support samples
        Fetch the first n_support samples as the support set per classes
        Return dtype: list
        """
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    
    def find_query_indxs(c):
        """
        Input a class 'c', return the indexes of query samples
        Return dtype: list
        """
        return target_cpu.eq(c).nonzero()[n_support:]

    # Get support and query set indexes
    support_indexes = list(map(find_support_idxs, classes))
    query_indexes = torch.stack(list(map(find_query_indxs, classes))).view(-1)

    # Compute prototype
    prototypes = torch.stack([prediction_cpu[idx_list].mean(0) for idx_list in support_indexes]) # idx_list is list with the same classes
    
    # Fetch the query sample predictions
    query_samples = prediction_cpu[query_indexes]

    # Compute distances
    dists = euclidean_dist(query_samples, prototypes) # [n_samples, n_classes], i.e. n_samples = n_classes * n_query

    log_prob = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1) # gegative



    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1) # log_prob的第一个维度是n_classes，所以才会转化成这样的形状
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_prob.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_prob.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val
